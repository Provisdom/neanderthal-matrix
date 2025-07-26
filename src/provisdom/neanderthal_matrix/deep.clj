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
(defn e-logistic!
  "Logistic: S(x) = e^(x) / [e^(x) + 1] = 0.5 + 0.5 * tanh(0.5 * x)."
  ([x]
   (e-axpb! 0.5 (e-tanh! (e-*! 0.5 x)) 0.5))
  ([x y]
   (e-axpb! 0.5 (e-tanh! (e-*! 0.5 (n-core/copy! x y))) 0.5)))

(defn e-logistic-derivative!
  "Logistic Derivative: S'(x) = S(x) * (1 - S(x))."
  [x]
  (with-release [x-raw (n-core/raw x)]
    (let [x (e-logistic! x)]
      (n-vect-math/mul! (e-axpb! -1.0 x 1.0 x-raw) x))))

(defn e-tanh-derivative!
  "tanh derivative: tanh(x)` = 1 - tanh(x)^2."
  [x]
  (with-release [x-raw (n-core/raw x)]
    (let [x (e-tanh! x)]
      (e-axpb! -1.0 (n-vect-math/mul! x x-raw) 1.0))))

(defn e-elu!
  "Exponential Linear Unit: f(x) = max(a * [exp^(x) - 1], x)."
  [alpha x]
  (n-vect-math/elu! alpha x))

(defn e-elu-derivative!
  "Exponential Linear Unit Derivative:
  f(x)` = {1 for x >= 0, a * exp^(x) for x <= 0}
        = {1 for x >= 0, f(x) + a for x <= 0}."
  [alpha x]
  ;;not obvious how to do this efficiently in Neanderthal
  )

(defn activation-e-elu!
  "Returns the elu function to be used as activation function."
  [alpha]
  (fn [x]
    (e-elu! alpha x)))

(defn e-relu!
  "Rectified Linear Unit: f(x) = max(0,x-T)."
  [threshold x]
  (n-vect-math/relu! threshold x))

(defn e-relu-derivative!
  "Rectified Linear Unit Derivative: f(x)` = {1 for x >= T, 0 for x < T}."
  [threshold x]
  ;;not obvious how to do this efficiently
  )

(defn activation-relu!
  "Returns the relu function to be used as activation function."
  [threshold]
  (fn [x]
    (e-relu! threshold x)))

;;;OBJECTIVE FUNCTIONS



;;;SANDBOX
(defn ->inference-layer
  ""
  [factory activation-fn in-dim out-dim]
  (let-release [weights-mx (n-core/ge factory out-dim in-dim)
                bias-v (n-core/vctr factory out-dim)]
    {:activation-fn activation-fn
     :bias-v        bias-v
     :invoke-fn     (fn [x ones-v a]
                      (activation-fn
                        (cross-product! -1.0 bias-v ones-v (n-core/mm! 1.0 weights-mx x 0.0 a))))
     :weights-mx    weights-mx}))

(defn ->training-layer
  ""
  ([inference-layer input-a-1 ones-v]
   (let [{:keys [activation-fn bias-v weights-mx]} inference-layer]
     (let-release [weights-mx* (n-core/view-ge weights-mx)  ;;was just "view"
                   bias-v* (n-core/view-vctr bias-v)
                   input-a-1* (n-core/view-ge input-a-1)
                   z (n-core/ge weights-mx (n-core/mrows weights-mx) (n-core/dim ones-v))
                   output-a (n-core/raw z)
                   ones-v* (n-core/view-vctr ones-v)]
       {:activation-fn activation-fn
        :bias-v        bias-v*
        :input-a       input-a-1*
        :forward-fn    (fn []
                         (activation-fn
                           (cross-product! -1.0 bias-v* ones-v* (n-core/mm! 1.0 weights-mx* input-a-1* 0.0 z))
                           output-a))
        :ones-v        ones-v*
        :output-a      output-a
        :weights-mx    weights-mx*})))
  ([inference-layer previous-training-layer]
   (->training-layer inference-layer
     (:output-a previous-training-layer)
     (:ones-v previous-training-layer))))

(defn this-particular-network
  ""
  [factory]
  (with-release [x (n-core/ge factory 2 2 [0.3 0.9 0.3 0.9])
                 ones-v (n-core/vctr factory 1 1)
                 layer-1 (->inference-layer factory e-tanh! 2 4)
                 layer-2 (->inference-layer factory e-logistic! 4 1)
                 training-layer-1 (->training-layer layer-1 x ones-v)
                 training-layer-2 (->training-layer layer-2 training-layer-1)]
    (n-core/transfer! [0.3 0.1 0.9 0.0 0.6 2.0 3.7 1.0] (:weights-mx layer-1))
    (n-core/transfer! [0.7 0.2 1.1 2.0] (:bias-v layer-1))
    (n-core/transfer! [0.75 0.15 0.22 0.33] (:weights-mx layer-2))
    (n-core/transfer! [0.3] (:bias-v layer-2))
    ((:forward-fn training-layer-1))
    ((:forward-fn training-layer-2))
    (n-core/transfer (:output-a training-layer-2))))
