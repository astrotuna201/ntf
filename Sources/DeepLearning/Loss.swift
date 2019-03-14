// Copyright 2018 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#if !COMPILING_TENSORFLOW_MODULE
import TensorFlow
#endif

import Netlib

///// Computes the mean squared error between logits and labels.
/////
///// - Parameters:
/////   - logits: One-hot encoded outputs from a neural network.
/////   - labels: One-hot encoded values that correspond to the correct output.
//@differentiable
//public func meanSquaredError<Scalar: TensorFlowFloatingPoint>(
//    predicted: TensorView<Scalar>, expected: TensorView<Scalar>
//) -> TensorView<Scalar> {
//    return (expected - predicted).squared().mean()
//}

///// Computes the softmax cross entropy (categorical cross entropy) between logits and labels.
/////
///// - Parameters:
/////   - logits: One-hot encoded outputs from a neural network.
/////   - labels: Indices (zero-indexed) of the correct outputs.
//@differentiable(wrt: logits, vjp: _vjpSoftmaxCrossEntropy)
//public func softmaxCrossEntropy<Scalar: TensorFlowFloatingPoint>(
//    logits: TensorView<Scalar>, labels: TensorView<Int32>
//) -> TensorView<Scalar> {
//    return Raw.sparseSoftmaxCrossEntropyWithLogits(features: logits, labels: labels).loss.mean()
//}
//
//@usableFromInline
//func _vjpSoftmaxCrossEntropy<Scalar: TensorFlowFloatingPoint>(
//    logits: TensorView<Scalar>, labels: TensorView<Int32>
//) -> (TensorView<Scalar>, (TensorView<Scalar>) -> TensorView<Scalar>) {
//    let (loss, grad) = Raw.sparseSoftmaxCrossEntropyWithLogits(features: logits, labels: labels)
//    return (loss.mean(), { v in v * grad })
//}
//
///// Computes the softmax cross entropy (categorical cross entropy) between logits and labels.
/////
///// - Parameters:
/////   - logits: One-hot encoded outputs from a neural network.
/////   - oneHotLabels: One-hot encoded values that correspond to the correct output.
//@differentiable
//public func softmaxCrossEntropy<Scalar: TensorFlowFloatingPoint>(
//    logits: TensorView<Scalar>, oneHotLabels: TensorView<Scalar>
//) -> TensorView<Scalar> {
//    return -(oneHotLabels * logSoftmax(logits)).mean(alongAxes: 0).sum()
//}
//
///// Computes the sigmoid cross entropy (binary cross entropy) between logits and labels.
/////
///// - Parameters:
/////   - logits: Single continuous values from `0` to `1`.
/////   - labels: Integer values that correspond to the correct output.
//@differentiable
//public func sigmoidCrossEntropy<Scalar: TensorFlowFloatingPoint>(
//    logits: TensorView<Scalar>, labels: TensorView<Scalar>
//) -> TensorView<Scalar> {
//    let loss = labels * log(logits) +
//        (TensorView<Scalar>(1) - labels) * log(TensorView<Scalar>(1) - logits)
//    return -loss.mean(alongAxes: 0).sum()
//}
