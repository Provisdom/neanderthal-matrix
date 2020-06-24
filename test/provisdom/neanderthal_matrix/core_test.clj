(ns provisdom.neanderthal-matrix.core-test
  (:require
    [clojure.test :refer :all]
    [clojure.spec.test.alpha :as st]
    [orchestra.spec.test :as ost]
    [criterium.core :as criterium]
    [provisdom.test.core :refer :all]
    [provisdom.math.core :as m]
    [provisdom.neanderthal-matrix.core :as core]))

;? seconds

(set! *warn-on-reflection* true)

(deftest lls-with-error-test
  (with-instrument [`core/lls-with-error]
    (is (spec-check core/lls-with-error))
    (is (data-approx=
          #::core
              {:standard-squared-errors [[3.3684210526315392]]
               :mean-squared-errors     [[1.6842105263157696]]
               :condition-number        6.499276998540634
               :annihilator             [[0.37969924812029987 -0.25187969924812054
                                          0.30075187969924816 -0.2857142857142859]
                                         [-0.25187969924812054 0.8007518796992481
                                          -0.12030075187969944 -0.2857142857142858]
                                         [0.30075187969924816 -0.12030075187969944
                                          0.24812030075187974 -0.285714285714286]
                                         [-0.2857142857142859 -0.2857142857142858
                                          -0.285714285714286 0.5714285714285714]]
               :solution                [[-0.6842105263157898] [3.473684210526316]]
               :projection              [[0.6203007518797001 0.25187969924812054
                                          -0.30075187969924816 0.2857142857142859]
                                         [0.25187969924812054 0.19924812030075195
                                          0.12030075187969944 0.2857142857142858]
                                         [-0.30075187969924816 0.12030075187969944
                                          0.7518796992481203 0.285714285714286]
                                         [0.2857142857142859 0.2857142857142858
                                          0.285714285714286 0.4285714285714286]]}
          (apply hash-map
                 (mapcat
                   (fn [[k v]]
                     (let [m (if (core/neanderthal-matrix? v)
                               (core/neanderthal-matrix->matrix v)
                               v)]
                       [k m]))
                   (core/lls-with-error
                     (core/matrix->neanderthal-matrix
                       [[1.0 2.0] [3.0 2.0] [6.0 2.0] [5.0 3.0]])
                     (core/matrix->neanderthal-matrix
                       [[5.0] [7.0] [2.0] [7.0]]))))))))

#_(ost/unstrument)