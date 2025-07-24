(ns provisdom.neanderthal-matrix.deep
  (:require
    [clojure.spec.alpha :as s]
    [clojure.spec.gen.alpha :as gen]
    [uncomplicate.clojurecl.core :as opencl :refer [*context* *command-queue* finish!]]
    [uncomplicate.commons.core :refer [let-release with-release]]
    [uncomplicate.fluokitten.core :refer [fmap!]]
    [uncomplicate.neanderthal.core :as n-core]
    [uncomplicate.neanderthal.math :as n-math]
    [uncomplicate.neanderthal.native :as n-native]
    ;[uncomplicate.neanderthal.opencl :as opencl]
    [uncomplicate.neanderthal.vect-math :as n-vect-math]))

(declare)

;;;REDEFINING
(def mv*! n-core/mv!)                                       ;; with 2-3 parameters X * y = z
(def e-max! n-vect-math/fmax!)                              ;; with 2-3 parameters max(x, y) = z
(def e-*! n-core/scal!)                                     ;; with 2-3 parameters a * x = y
(def e-axpb! n-vect-math/linear-frac!)                      ;; with 3-4 parameters: a * x + b = y
(def e-linear-fraction! n-vect-math/linear-frac!)           ;; with 6-7 parameters: (a * x + b) / (c * y + d) = z
(def e-axpy! n-core/axpy!)
(def e-axpby! n-core/axpby!)
(def e-tanh! n-vect-math/tanh!)                             ;;Hyperbolic Tangent: f(x) = sinh(x) / cosh(x) = [e^(2x) - 1] /[ e^(2x) + 1].
(def cross-product! n-core/rk!)                             ;; with 1 or 2 parameters tanh(x) = y

;;;ACTIVATION FUNCTIONS

;;e-tanh! makes a good activation function

(defn e-relu!
  "Rectified Linear Unit: f(x) = max(0,x)."
  [threshold-v x]
  (n-core/axpy! -1.0 threshold-v (e-max! threshold-v x x)))

(defn activation-relu!
  "Returns the relu function to be used as activation function."
  [threshold-v]
  (fn [x]
    (e-relu! threshold-v x)))

(defn e-sigmoid!
  "Sigmoid: S(x) = e^(x) / [e^(x) + 1] = 0.5 + 0.5 * tanh(0.5 * x)."
  [x]
  (e-axpb! 0.5 (e-tanh! (e-*! 0.5 x)) 0.5))

;;;SANDBOX
(defn ->fully-connected-inference
  ""
  [factory activation-fn in-dim out-dim]
  (let-release [weights-mx (n-core/ge factory out-dim in-dim)
                bias-v (n-core/vctr factory out-dim)]
    {:bias-v     bias-v
     :invoke-fn  (fn [x ones a]
                   (activation-fn
                     (cross-product! -1.0 bias-v ones (n-core/mm! 1.0 weights-mx x 0.0 a))))
     :weights-mx weights-mx}))

(defn this-particular-network
  ""
  [factory]
  (with-release [x (n-core/ge factory 2 2 [0.3 0.9 0.3 0.9])
                 ones (n-core/vctr factory 1 1)
                 layer-1 (->fully-connected-inference factory e-tanh! 2 4)
                 a-1 (n-core/ge factory 4 2)
                 layer-2 (->fully-connected-inference factory e-sigmoid! 4 1)
                 a-2 (n-core/ge factory 1 2)]
    (n-core/transfer! [0.3 0.1 0.9 0.0 0.6 2.0 3.7 1.0] (:weights-mx layer-1))
    (n-core/transfer! [0.7 0.2 1.1 2.0] (:bias-v layer-1))
    (n-core/transfer! [0.75 0.15 0.22 0.33] (:weights-mx layer-2))
    (n-core/transfer! [0.3] (:bias-v layer-2))
    (n-core/transfer ((:invoke-fn layer-2) ((:invoke-fn layer-1) x ones a-1) ones a-2))))
