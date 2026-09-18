"""Small AlphaZero-style policy/value network (TensorFlow / Keras).

REPAIR NOTES (relative to the original repository)
--------------------------------------------------
The original ``NeuralNetwork``:

  * tied the policy head width to the *input* width
    (``Dense(input_size, activation='softmax')``), so a flat board vector of
    768 features implied a 768-way policy -- not a chess action space;
  * emitted a softmax policy directly, which makes illegal-move masking and a
    numerically stable cross-entropy awkward;
  * had ``train()`` as ``pass`` (a placeholder, no optimiser, no losses);
  * defined ``save_weights``/``load_weights`` that called ``self.save`` /
    ``self.load`` -- ``load`` does not exist on ``tf.keras.Model`` and the
    override shadowed Keras' own weight API, so checkpointing was broken.

What is kept: the class name ``NeuralNetwork``, its ``tf.keras.Model`` base, the
two-head (policy, value) design, and the ``predict`` / ``train`` /
``save_weights`` / ``load_weights`` public surface.

What changed: a small configurable convolutional trunk over the (8, 8, 20)
observation, **policy logits** (masked downstream) of size ``NUM_ACTIONS``, a
``tanh``-bounded scalar value for the player to move, a real training step with
masked cross-entropy + optional masked value regression, batched inference, and
checkpoints that store architecture metadata, weights, optimiser state, RNG
state and the encoding version.
"""

from __future__ import annotations

import json
import os
import pickle

import numpy as np
import tensorflow as tf

from envs.chess_utils import ENCODING_VERSION, NUM_ACTIONS, OBS_SHAPE, OBS_VERSION

DEFAULT_ARCH = {
    "trunk": "conv",
    "channels": 64,
    "blocks": 4,
    "policy_channels": 4,
    "value_hidden": 64,
    "obs_shape": list(OBS_SHAPE),
    "num_actions": NUM_ACTIONS,
    "encoding_version": ENCODING_VERSION,
    "obs_version": OBS_VERSION,
}


class NeuralNetwork(tf.keras.Model):
    """Compact residual-conv policy/value network.

    ``input_size`` is accepted for backwards compatibility with the original
    constructor signature but the observation shape comes from ``arch``.
    """

    def __init__(self, input_size=None, arch: dict | None = None, **kwargs):
        super().__init__(**kwargs)
        self.arch = dict(DEFAULT_ARCH)
        if arch:
            self.arch.update(arch)
        c = self.arch["channels"]
        self.obs_shape = tuple(self.arch["obs_shape"])
        self.num_actions = int(self.arch["num_actions"])

        reg = tf.keras.regularizers.l2(1e-4)
        self.stem = tf.keras.layers.Conv2D(c, 3, padding="same", use_bias=False, kernel_regularizer=reg)
        self.stem_bn = tf.keras.layers.BatchNormalization()
        self.res_convs = []
        for _ in range(int(self.arch["blocks"])):
            self.res_convs.append(
                (
                    tf.keras.layers.Conv2D(c, 3, padding="same", use_bias=False, kernel_regularizer=reg),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Conv2D(c, 3, padding="same", use_bias=False, kernel_regularizer=reg),
                    tf.keras.layers.BatchNormalization(),
                )
            )

        pc = int(self.arch["policy_channels"])
        self.policy_conv = tf.keras.layers.Conv2D(pc, 1, padding="same", use_bias=False, kernel_regularizer=reg)
        self.policy_bn = tf.keras.layers.BatchNormalization()
        self.policy_flat = tf.keras.layers.Flatten()
        # Policy LOGITS (not softmax): illegal actions are masked before the
        # softmax is taken, which the original softmax head made impossible.
        self.policy_output = tf.keras.layers.Dense(self.num_actions, kernel_regularizer=reg, name="policy_logits")

        self.value_conv = tf.keras.layers.Conv2D(1, 1, padding="same", use_bias=False, kernel_regularizer=reg)
        self.value_bn = tf.keras.layers.BatchNormalization()
        self.value_flat = tf.keras.layers.Flatten()
        self.value_hidden = tf.keras.layers.Dense(int(self.arch["value_hidden"]), activation="relu", kernel_regularizer=reg)
        self.value_output = tf.keras.layers.Dense(1, activation="tanh", kernel_regularizer=reg, name="value")

        self.optimizer_ = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self._infer_fn = None
        # Force a deterministic build of every sublayer (and therefore a stable
        # ``self.weights`` ordering) with one dummy forward pass.  Calling
        # ``self.build(...)`` is not enough in Keras 3: sublayers stay unbuilt,
        # ``self.weights`` comes back partial, and checkpoint save/load then
        # silently mismatches because ``zip`` truncates.
        self(tf.zeros((1,) + self.obs_shape, dtype=tf.float32), training=False)

    # ----------------------------------------------------------------- forward
    def call(self, inputs, training=False, return_hidden=False):
        x = tf.nn.relu(self.stem_bn(self.stem(inputs), training=training))
        for c1, b1, c2, b2 in self.res_convs:
            y = tf.nn.relu(b1(c1(x), training=training))
            y = b2(c2(y), training=training)
            x = tf.nn.relu(x + y)
        hidden = x  # exposed intermediate representation for later probing

        p = tf.nn.relu(self.policy_bn(self.policy_conv(x), training=training))
        policy_logits = self.policy_output(self.policy_flat(p))

        v = tf.nn.relu(self.value_bn(self.value_conv(x), training=training))
        value = self.value_output(self.value_hidden(self.value_flat(v)))
        value = tf.squeeze(value, axis=-1)

        if return_hidden:
            return policy_logits, value, hidden
        return policy_logits, value

    # --------------------------------------------------------------- inference
    def _get_infer_fn(self):
        if self._infer_fn is None:
            @tf.function(reduce_retracing=True)
            def _fn(x):
                return self(x, training=False)

            self._infer_fn = _fn
        return self._infer_fn

    def predict(self, state):
        """Batched inference.  ``state`` may be (8,8,20) or (B,8,8,20).

        Returns ``(policy_logits, value)`` as numpy arrays.  Inference mode is
        applied (BatchNorm uses running statistics).
        """
        arr = np.asarray(state, dtype=np.float32)
        single = arr.ndim == 3
        if single:
            arr = arr[None, ...]
        logits, value = self._get_infer_fn()(tf.convert_to_tensor(arr))
        logits, value = logits.numpy(), value.numpy()
        if single:
            return logits[0], float(value[0])
        return logits, value

    def hidden_representation(self, state):
        arr = np.asarray(state, dtype=np.float32)
        single = arr.ndim == 3
        if single:
            arr = arr[None, ...]
        _, _, h = self(tf.convert_to_tensor(arr), training=False, return_hidden=True)
        h = h.numpy()
        return h[0] if single else h

    # ---------------------------------------------------------------- training
    @tf.function(reduce_retracing=True)
    def _train_step(self, obs, target_pi, legal_mask, target_v, v_weight, value_coef):
        with tf.GradientTape() as tape:
            logits, value = self(obs, training=True)
            neg_inf = tf.constant(-1e9, dtype=logits.dtype)
            masked = tf.where(legal_mask, logits, neg_inf)
            logp = tf.nn.log_softmax(masked, axis=-1)
            policy_loss = -tf.reduce_mean(tf.reduce_sum(target_pi * logp, axis=-1))
            sq = tf.square(value - target_v)
            denom = tf.maximum(tf.reduce_sum(v_weight), 1.0)
            value_loss = tf.reduce_sum(sq * v_weight) / denom
            reg_loss = tf.add_n(self.losses) if self.losses else tf.constant(0.0)
            loss = policy_loss + value_coef * value_loss + reg_loss
        grads = tape.gradient(loss, self.trainable_variables)
        grads, gnorm = tf.clip_by_global_norm(grads, 5.0)
        self.optimizer_.apply_gradients(zip(grads, self.trainable_variables))
        return loss, policy_loss, value_loss, gnorm

    def train(self, data, value_coef: float = 1.0):
        """One gradient step on a minibatch.

        ``data`` is a dict with keys ``obs`` (B,8,8,20), ``pi`` (B,A) target
        distribution, ``mask`` (B,A) legal mask, ``v`` (B,) value target and
        ``v_weight`` (B,) in {0,1}.  A weight of 0 means the game-theoretic
        value is **unknown** and the value loss is masked out for that sample --
        unknown outcomes are never silently labelled as draws.
        """
        obs = tf.convert_to_tensor(np.asarray(data["obs"], dtype=np.float32))
        pi = tf.convert_to_tensor(np.asarray(data["pi"], dtype=np.float32))
        mask = tf.convert_to_tensor(np.asarray(data["mask"], dtype=bool))
        v = tf.convert_to_tensor(np.asarray(data["v"], dtype=np.float32))
        vw = tf.convert_to_tensor(np.asarray(data["v_weight"], dtype=np.float32))
        loss, pl, vl, gnorm = self._train_step(
            obs, pi, mask, v, vw, tf.constant(float(value_coef), dtype=tf.float32)
        )
        return {
            "loss": float(loss.numpy()),
            "policy_loss": float(pl.numpy()),
            "value_loss": float(vl.numpy()),
            "grad_norm": float(gnorm.numpy()),
        }

    def set_learning_rate(self, lr: float):
        self.optimizer_.learning_rate.assign(lr)

    # ------------------------------------------------------------ checkpoints
    def save_weights(self, path, extra: dict | None = None):
        """Save architecture metadata + weights + optimiser state + RNG state."""
        path = str(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        meta = {
            "arch": self.arch,
            "encoding_version": ENCODING_VERSION,
            "obs_version": OBS_VERSION,
            "extra": extra or {},
        }
        with open(path + ".meta.json", "w") as fh:
            json.dump(meta, fh, indent=2)
        payload = {
            "weights": [w.numpy() for w in self.weights],
            "optimizer": [v.numpy() for v in self.optimizer_.variables],
            "numpy_rng_state": np.random.get_state(),
        }
        with open(path + ".pkl", "wb") as fh:
            pickle.dump(payload, fh, protocol=4)
        return path

    def load_weights(self, path, restore_optimizer: bool = False):
        path = str(path)
        with open(path + ".pkl", "rb") as fh:
            payload = pickle.load(fh)
        saved = payload["weights"]
        if len(saved) != len(self.weights):
            raise ValueError(
                f"checkpoint has {len(saved)} weight tensors but this model has "
                f"{len(self.weights)}; architecture metadata mismatch"
            )
        for w, arr in zip(self.weights, saved):
            if tuple(w.shape) != tuple(np.shape(arr)):
                raise ValueError(f"weight shape mismatch: {tuple(w.shape)} vs {np.shape(arr)}")
            w.assign(arr)
        if restore_optimizer and payload.get("optimizer"):
            # Force optimiser variable creation before assignment.
            if not self.optimizer_.variables:
                self.optimizer_.build(self.trainable_variables)
            for v, arr in zip(self.optimizer_.variables, payload["optimizer"]):
                try:
                    v.assign(arr)
                except Exception:
                    pass
        self._infer_fn = None
        return self

    @classmethod
    def from_checkpoint(cls, path, restore_optimizer: bool = False) -> "NeuralNetwork":
        with open(str(path) + ".meta.json") as fh:
            meta = json.load(fh)
        model = cls(arch=meta["arch"])
        model.load_weights(path, restore_optimizer=restore_optimizer)
        return model

    def parameter_count(self) -> int:
        return int(sum(int(np.prod(w.shape)) for w in self.weights))
