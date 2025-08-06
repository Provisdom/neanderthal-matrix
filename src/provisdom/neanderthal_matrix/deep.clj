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
;;every function is quite overloaded
(def mv*! n-core/mv!)                                       ;; with 2-5 parameters: aX * y + bz = w
(def e-*! n-core/scal!)                                     ;; with 2 parameters: a * x = y
(def e-axpby! n-core/axpby!)
(def mm*! n-core/mm!)
(def cross-product! n-core/rk!)                             ;;with 2-4 parameters a[x ✕ y] + B

(def e-axpb! n-vect-math/linear-frac!)                      ;; with 2-4 parameters: a * x + b = y
(def e-linear-fraction! n-vect-math/linear-frac!)           ;; with 7 parameters: (a * x + b) / (c * y + d) = z
(def e-axpy! n-core/axpy!)

;;many n-vect-math functions aren't recognized IntelliJ
(def e-max! n-vect-math/fmax!)                              ;; with 2-3 parameters: max(x, y) = z
(def e-mul! n-vect-math/mul!)
(def e-sq! n-vect-math/sqr!)
(def e-sq n-vectomath/sqr)
(def e-inv! n-vect-math/inv!)
(def e-cosh! n-vect-math/cosh!)

;;Hyperbolic Tangent: f(x) = sinh(x) / cosh(x) = [e^(2x) - 1] /[ e^(2x) + 1].
(def e-tanh! n-vect-math/tanh!)                             ;; with 1 or 2 parameters tanh(x) = y

;;;ACTIVATION FUNCTIONS
(defn e-logistic!
  "Logistic: S(x) = e^(x) / [e^(x) + 1] = 0.5 + 0.5 * tanh(0.5 * x)."
  ([x!]
   (e-axpb! 0.5 (e-tanh! (e-*! 0.5 x!)) 0.5))
  ([x y!]
   (e-axpb! 0.5 (e-tanh! (e-*! 0.5 (n-core/copy! x y!))) 0.5)))

(defn e-logistic-derivative!
  "Logistic Derivative: S'(x) = S(x) * (1 - S(x))."
  [x!]
  (with-release [x-raw (n-core/raw x!)]
    (let [x! (e-logistic! x!)]
      (e-mul! (e-axpb! -1.0 x! 1.0 x-raw) x!))))

(defn e-tanh-derivative!
  "tanh derivative: tanh(x)` = 1 - tanh(x)^2 = cosh(x)^-2."
  [x!]
  (e-sq! (e-inv! (e-cosh! x!))))

(defn e-elu!
  "Exponential Linear Unit: f(x) = max(a * [exp^(x) - 1], x)."
  [alpha x!]
  (n-vect-math/elu! alpha x!))

(defn e-elu-derivative!
  "Exponential Linear Unit Derivative:
  f(x)` = {1 for x >= 0, a * exp^(x) for x <= 0}
        = {1 for x >= 0, f(x) + a for x <= 0}."
  [alpha x!]
  ;;not obvious how to do this efficiently in Neanderthal
  )

(defn activation-e-elu!
  "Returns the elu function to be used as activation function."
  [alpha]
  (fn [x!]
    (e-elu! alpha x!)))

(defn e-relu!
  "Rectified Linear Unit: f(x) = max(0,x-T)."
  [threshold x!]
  (n-vect-math/relu! threshold x!))

(defn e-relu-derivative!
  "Rectified Linear Unit Derivative: f(x)` = {1 for x >= T, 0 for x < T}."
  [threshold x!]
  ;;not obvious how to do this efficiently
  )

(defn activation-relu!
  "Returns the relu function to be used as activation function."
  [threshold]
  (fn [x!]
    (e-relu! threshold x!)))

;;;OBJECTIVE FUNCTIONS
(defn mean-squared-error!
  [diff-mx]
  (/ (e-sq (n-core/nrm2 diff-mx)) (n-core/dim diff-mx)))

(defn mean-squared-error-derivative!
  ([diff-mx!]
   (n-core/scal! (/ 2.0 (n-core/dim diff-mx!)) diff-mx!))
  ([target-mx output-a!]
   (let [s (/ 2.0 (n-core/dim target-mx))]
     (n-core/axpby! (* -1.0 s) target-mx s output-a!))))

;;;NEURAL NETWORK
(defn this-particular-network
  ""
  [factory]
  (let [eta 0.05]
    (with-release [x (n-core/ge factory 2 2 [0.3 0.9 0.3 0.9])
                   ones-v (n-core/vctr factory 1 1)         ;2 dim -- can get this from x-dim
                   layer-1 (->inference-layer factory e-tanh-derivative! e-tanh! 2 4)
                   layer-2 (->inference-layer factory e-logistic-derivative! e-logistic! 4 1)
                   training-layer-1 (->training-layer layer-1 x ones-v)
                   training-layer-2 (->training-layer layer-2 training-layer-1)]
      (n-core/transfer! [0.3 0.1 0.9 0.0 0.6 2.0 3.7 1.0] (:weights-mx layer-1))
      (n-core/transfer! [0.7 0.2 1.1 2.0] (:bias-v layer-1))
      (n-core/transfer! [0.75 0.15 0.22 0.33] (:weights-mx layer-2))
      (n-core/transfer! [0.3] (:bias-v layer-2))
      ((:forward-fn! training-layer-1))
      ((:forward-fn! training-layer-2))
      ((:backward-fn! training-layer-2) eta)
      ((:backward-fn! training-layer-1) eta)
      (n-core/transfer (:output-a training-layer-2)))))

;;input-data is in,data
;;W is out,in (weights-mx)
;;B is out (bias-v)
;;weighted-output is out,data (W*X+B)
;;output-a is out,data f(weighted-output)
;;input-a is in,data
;;temp-w-mx is out,in (used for temp storage)

(defn feedforward
  "Docstring"
  [activation-fn! weights-mx bias-v ones-v input-a weighted-output! output-a!]
  ;put W*X into weighted-output! (0.0 means add zero of weighted-output!)
  (n-core/mm! 1.0 weights-mx input-a 0.0 weighted-output!)
  ;put W*X-B into weighted-output!
  (cross-product! -1.0 bias-v ones-v weighted-output!)
  ;put f(W*X-B) into output-a!
  (activation-fn! weighted-output! output-a!))

(defn backpropagate
  "Only re-use values from weights-mx!, bias-v!, and input-a! out of this function.
  The remainder of the changed outputs are just memory allocations."
  [activation-derivative-fn! eta-learning-parameter input-a! output-a ones-v weighted-output! temp-w-mx! weights-mx! bias-v!]
  (let [eta-avg (- (/ (double eta-learning-parameter) (n-core/dim ones-v)))]
    ;;put f`(old-z) into weighted-output!
    (activation-derivative-fn! weighted-output!)
    ;;put f`(old-z) ⨷ ([W]^T * next-d) into weighted-output! => output-error d
    (e-mul! weighted-output! output-a)
    ;;put -e/n * d * [input-a-1]^T into temp-w-mx!
    (n-core/mm! eta-avg weighted-output! (n-core/trans input-a!) 0.0 temp-w-mx!)
    ;;put [W]^T * d into input-a! => weighted output error
    (n-core/mm! 1.0 (n-core/trans weights-mx!) weighted-output! 0.0 input-a!)
    ;;put -e/n * SUMi[d] + old-b into b! => new b
    (n-core/mv! eta-avg weighted-output! ones-v 1.0 bias-v!)
    ;;put -e/n * d * [input-a-1]^T + W-old into W! => new W
    (e-axpy! 1.0 temp-w-mx! weights-mx!)))

(s/def ::network-layer
  (s/keys :req [::dim ::activation-derivative-fn ::activation-fn]))

(s/def ::network-layers
  (s/coll-of ::network-layer))                              ;;TODO: turn into vector with min-count

(s/def ::connected-network-layer                            ;;out-dim is mrows for weights-mx
  (s/with-gen
    (s/and (s/merge ::network-layer
             (s/keys :req [::bias-v ::weights-mx]))
      (fn [{::keys [bias-v dim weights-mx]}]
        (= (n-core/mrows weights-mx) (n-core/dim bias-v) dim)))
    #(gen/generate (s/gen ::connected-network-layer))))     ;;;TODO: write generator

(s/def ::neural-network
  (s/coll-of ::connected-network-layer))                    ;;TODO: turn into vector with min-count
;;;TODO: ensure that all in-dim (ncols) for weights-mx matches dim for previous layer

(s/def ::batch-connected-network-layer
  (s/with-gen
    (s/and (s/merge ::connected-network-layer
             (s/keys :req [::output-a ::weighted-output]))
      (fn []))
    #()))

(defn ->connected-network-layer
  "Docstring"
  [factory in-dim network-layer]
  (let [{::keys [dim]} network-layer]
    (with-release [bias-v (n-core/vctr factory dim)
                   weights-mx (n-core/ge factory dim in-dim)]
      (assoc network-layer
        ::bias-v bias-v
        ::weights-mx weights-mx))))

(defn ->neural-network
  "Docstring"
  [factory input-dim network-layers]
  (reduce
    (fn [acc network-layer]
      (let [in-dim (or (::dim (last acc)) input-dim)]
        (conj acc (->connected-network-layer factory in-dim network-layer))))
    []
    network-layers))

(defn ->batch-connected-network-layer
  "Docstring"
  [connected-network-layer data-dim]
  (let [{::keys [weights-mx]} connected-network-layer
        out-dim (n-core/mrows weights-mx)]
    (with-release [output-a (n-core/ge weights-mx out-dim data-dim)
                   weighted-output (n-core/ge weights-mx out-dim data-dim)]
      (assoc connected-network-layer
        ::output-a output-a
        ::weighted-output weighted-output))))

(defn ->ones-v
  "Docstring"
  [input-data]
  (with-release [ones-v (n-core/entry! (n-core/vctr input-data (n-core/ncols input-data)) 1.0)]
    ones-v))

(defn ->weighted-output
  "Docstring"
  [ones-v weights-mx]
  (with-release [weighted-output (n-core/ge weights-mx
                                   (n-core/mrows weights-mx)
                                   (n-core/dim ones-v))]
    weighted-output))

(defn run-neural-network-epoch
  "Docstring"
  ([eta-learning-parameter obj-fn neural-network input-data]
   (run-neural-network-epoch eta-learning-parameter obj-fn neural-network input-data (->ones-v input-data)))
  ([eta-learning-parameter obj-fn neural-network input-data ones-v]
   ;;ensure x and ones-v sizes match with spec
   (let [data-dim (n-core/ncols input-data)
         batch-neural-network (mapv #(->batch-connected-network-layer % data-dim)
                                neural-network)
         forward-nn (reduce (fn [acc batch-connected-network-layer]

                   ))
         (feedforward activation-fn! weights-mx bias-v ones-v input-a weighted-output! output-a!)
         ]

     ;;as it's built forward, create the '::batch-connected-network-layers'
     ;; as it goes backward, create and use a `::temp-w-mx` or two.
     ;;  then convert the batch-connected... back into connected-network-layers as we go (releasing memory)
     ;;;TODO: may need to handle cases of:
     ;;; (1) one layer -- probably no need for extra memory
     ;;; (2) two layers -- need one temp object
     ;;; (3) 3+ layers -- need two temp objects (alternating)
     )))

;;;TODO: (1) Don’t need to update input-a if the first layer; need desired-output
;;; (2) Bayadera for GPU random numbers (see 13)
;;; (3) Don’t build layers by just adding to maps —
;;;        need to use let-release, and use with-release around it
;;;        (constructor can take the more narrow version — and have a function
;;;        that returns the more narrow version
