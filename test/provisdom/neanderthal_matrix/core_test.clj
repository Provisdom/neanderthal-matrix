(ns provisdom.neanderthal-matrix.core-test
  (:require
    [clojure.spec.test.alpha :as st]
    [clojure.test :refer :all]
    [provisdom.test.core :refer :all]
    [provisdom.neanderthal-matrix.core :as neanderthal]))

;? seconds

(set! *warn-on-reflection* true)

(deftest lls-with-error-test
  (with-instrument `neanderthal/lls-with-error
    (is (spec-check neanderthal/lls-with-error)))
  (with-instrument (st/instrumentable-syms)
    (is=
      #::neanderthal
              {:annihilator
               [[0.37969924812029987 -0.25187969924812054 0.30075187969924816
                 -0.2857142857142859]
                [-0.25187969924812054 0.8007518796992481 -0.12030075187969944
                 -0.2857142857142858]
                [0.30075187969924816 -0.12030075187969944 0.24812030075187974
                 -0.285714285714286]
                [-0.2857142857142859 -0.2857142857142858 -0.285714285714286
                 0.5714285714285714]]

               :condition-number        6.499276998540634
               :mean-squared-errors     [[1.6842105263157696]]

               :projection
               [[0.6203007518797001 0.25187969924812054 -0.30075187969924816
                 0.2857142857142859]
                [0.25187969924812054 0.19924812030075195 0.12030075187969944
                 0.2857142857142858]
                [-0.30075187969924816 0.12030075187969944 0.7518796992481203
                 0.285714285714286]
                [0.2857142857142859 0.2857142857142858 0.285714285714286
                 0.4285714285714286]]

               :solution
               [[-0.6842105263157898] [3.473684210526316]]

               :standard-squared-errors [[3.3684210526315392]]}
      (apply hash-map
        (mapcat
          (fn [[k v]]
            (let [m (if (neanderthal/neanderthal-matrix? v)
                      (neanderthal/neanderthal-matrix->matrix v)
                      v)]
              [k m]))
          (neanderthal/lls-with-error
            (neanderthal/matrix->neanderthal-matrix
              [[1.0 2.0] [3.0 2.0] [6.0 2.0] [5.0 3.0]])
            (neanderthal/matrix->neanderthal-matrix
              [[5.0] [7.0] [2.0] [7.0]])))))))
