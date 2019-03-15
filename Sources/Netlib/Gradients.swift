////===-- Gradients.swift ---------------------------------------*- swift -*-===//
////
//// This source file is part of the Swift.org open source project
////
//// Copyright (c) 2014 - 2017 Apple Inc. and the Swift project authors
//// Licensed under Apache License v2.0 with Runtime Library Exception
////
//// See https://swift.org/LICENSE.txt for license information
//// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
////
////===----------------------------------------------------------------------===//
////
//// This file contains vector-Jacobian product (VJP) definitions for TensorView ops.
////
//// Terminology:
//// - originalValue (f): The function being differentiated, or the result of that
////   function.
//// - VJP (f'): The function as the result of differentiation, computing
////   the vector-Jacobian products with respect to all arguments, or the result
////   of that function.
////
//// For more information, visit:
//// https://en.wikipedia.org/wiki/Automatic_differentiation
////
//// Every function in this file is the VJP of some corresponding function
//// defined in Ops.swift, with respect to all arguments. The attribute
//// '@differentiable(vjp: ...)' is used to register a function's VJP. The
//// automatic differentiation pass identifies these VJPs and chains them
//// together to produce arbitrary differentiable programs.
////
//// NOTE:
//// - Currently, we do not want to expose VJP functions to users. The name of
////   each VJP function should start with an underscore.
////
//// TODO:
//// - Fix VJPs for broadcasting ops (need to perform reduction).
////
////===----------------------------------------------------------------------===//
//
//infix operator .== : ComparisonPrecedence
//infix operator .> : ComparisonPrecedence
//
////===----------------------------------------------------------------------===//
//// Method-style differential operators
////===----------------------------------------------------------------------===//
//
//public extension Differentiable {
//  @inlinable
//  func gradient<R : TensorFlowFloatingPoint>(
//    in f: @differentiable (Self) -> TensorView<R>
//  ) -> CotangentVector {
//    return self.pullback(in: f)(TensorView<R>(1))
//  }
//
//  @inlinable
//  func valueWithGradient<R : TensorFlowFloatingPoint>(
//    in f: @differentiable (Self) -> TensorView<R>
//  ) -> (value: TensorView<R>, gradient: CotangentVector) {
//    let (y, pb) = self.valueWithPullback(in: f)
//    return (y, pb(TensorView<R>(1)))
//  }
//
//  @inlinable
//  func gradient<T : Differentiable, R : TensorFlowFloatingPoint>(
//    at x: T, in f: @differentiable (Self, T) -> TensorView<R>
//  ) -> (CotangentVector, T.CotangentVector) {
//    return self.pullback(at: x, in: f)(TensorView<R>(1))
//  }
//
//  @inlinable
//  func valueWithGradient<T : Differentiable, R : TensorFlowFloatingPoint>(
//    at x: T, in f: @differentiable (Self, T) -> TensorView<R>
//  ) -> (value: TensorView<R>, gradient: (CotangentVector, T.CotangentVector)) {
//    let (y, pb) = self.valueWithPullback(at: x, in: f)
//    return (y, pb(TensorView<R>(1)))
//  }
//}
//
////===----------------------------------------------------------------------===//
//// Free-function-style differential operators
////===----------------------------------------------------------------------===//
//
//// Value with gradient
//
//@inlinable
//public func valueWithGradient<T, R>(
//  at x: T, in f: @differentiable (T) -> TensorView<R>
//) -> (value: TensorView<R>, gradient: T.CotangentVector)
//where T : Differentiable, R : TensorFlowFloatingPoint {
//  let (y, pullback) = valueWithPullback(at: x, in: f)
//  return (y, pullback(TensorView<R>(1)))
//}
//
//@inlinable
//public func valueWithGradient<T, U, R>(
//  at x: T, _ y: U, in f: @differentiable (T, U) -> TensorView<R>
//) -> (value: TensorView<R>, gradient: (T.CotangentVector, U.CotangentVector))
//  where T : Differentiable, U : Differentiable,
//        R : TensorFlowFloatingPoint {
//  let (y, pullback) = valueWithPullback(at: x, y, in: f)
//  return (y, pullback(TensorView<R>(1)))
//}
//
//@inlinable
//public func valueWithGradient<T, U, V, R>(
//  at x: T, _ y: U, _ z: V, in f: @differentiable (T, U, V) -> TensorView<R>
//) -> (value: TensorView<R>,
//      gradient: (T.CotangentVector, U.CotangentVector, V.CotangentVector))
//  where T : Differentiable, U : Differentiable, V : Differentiable,
//        R : TensorFlowFloatingPoint {
//  let (y, pullback) = valueWithPullback(at: x, y, z, in: f)
//  return (y, pullback(TensorView<R>(1)))
//}
//
//// Value with gradient (curried)
//
//@inlinable
//public func valueWithGradient<T, R>(
//  of f: @escaping @differentiable (T) -> TensorView<R>
//) -> (T) -> (value: TensorView<R>, gradient: T.CotangentVector)
//  where T : Differentiable, R : TensorFlowFloatingPoint {
//  return { x in valueWithGradient(at: x, in: f) }
//}
//
//@inlinable
//public func valueWithGradient<T, U, R>(
//  of f: @escaping @differentiable (T, U) -> TensorView<R>
//) -> (T, U)
//    -> (value: TensorView<R>, gradient: (T.CotangentVector, U.CotangentVector))
//  where T : Differentiable, U : Differentiable,
//        R : TensorFlowFloatingPoint {
//  return { x, y in valueWithGradient(at: x, y, in: f) }
//}
//
//@inlinable
//public func valueWithGradient<T, U, V, R>(
//  of f: @escaping @differentiable (T, U, V) -> TensorView<R>
//) -> (T, U, V)
//    -> (value: TensorView<R>,
//        gradient: (T.CotangentVector, U.CotangentVector, V.CotangentVector))
//  where T : Differentiable, U : Differentiable, V : Differentiable,
//        R : TensorFlowFloatingPoint {
//  return { x, y, z in valueWithGradient(at: x, y, z, in: f) }
//}
//
//// Gradient
//
//@inlinable
//public func gradient<T, R>(
//  at x: T, in f: @differentiable (T) -> TensorView<R>
//) -> T.CotangentVector
//  where T : Differentiable, R : TensorFlowFloatingPoint {
//  return pullback(at: x, in: f)(TensorView<R>(1))
//}
//
//@inlinable
//public func gradient<T, U, R>(
//  at x: T, _ y: U, in f: @differentiable (T, U) -> TensorView<R>
//) -> (T.CotangentVector, U.CotangentVector)
//  where T : Differentiable, U : Differentiable,
//        R : TensorFlowFloatingPoint {
//  return pullback(at: x, y, in: f)(TensorView<R>(1))
//}
//
//@inlinable
//public func gradient<T, U, V, R>(
//  at x: T, _ y: U, _ z: V, in f: @differentiable (T, U, V) -> TensorView<R>
//) -> (T.CotangentVector, U.CotangentVector, V.CotangentVector)
//  where T : Differentiable, U : Differentiable, V : Differentiable,
//        R : TensorFlowFloatingPoint {
//  return pullback(at: x, y, z, in: f)(TensorView<R>(1))
//}
//
//// Gradient (curried)
//
//@inlinable
//public func gradient<T, R>(
//  of f: @escaping @differentiable (T) -> TensorView<R>
//) -> (T) -> T.CotangentVector
//  where T : Differentiable, R : TensorFlowFloatingPoint {
//  return { x in gradient(at: x, in: f) }
//}
//
//@inlinable
//public func gradient<T, U, R>(
//  of f: @escaping @differentiable (T, U) -> TensorView<R>
//) -> (T, U) -> (T.CotangentVector, U.CotangentVector)
//  where T : Differentiable, U : Differentiable,
//        R : TensorFlowFloatingPoint {
//  return { x, y in gradient(at: x, y, in: f) }
//}
//
//@inlinable
//public func gradient<T, U, V, R>(
//  of f: @escaping @differentiable (T, U, V) -> TensorView<R>
//) -> (T, U, V) -> (T.CotangentVector, U.CotangentVector, V.CotangentVector)
//  where T : Differentiable, U : Differentiable, V : Differentiable,
//        R : TensorFlowFloatingPoint {
//  return { x, y, z in gradient(at: x, y, z, in: f) }
//}
//
////===----------------------------------------------------------------------===//
//// Elementwise binary
////===----------------------------------------------------------------------===//
//
//extension TensorView where Scalar : TensorFlowFloatingPoint {
//  @inlinable
//  static func _vjpAdd(
//    lhs: TensorView, rhs: TensorView
//  ) -> (TensorView, (TensorView) -> (TensorView, TensorView)) {
//    return (lhs + rhs, {
//      [lhsShape = lhs.shapeTensor, rhsShape = rhs.shapeTensor] v in
//      return (v.unbroadcast(toShape: lhsShape), v.unbroadcast(toShape: rhsShape))
//    })
//  }
//
//  @inlinable
//  static func _vjpSubtract(
//    lhs: TensorView, rhs: TensorView
//  ) -> (TensorView, (TensorView) -> (TensorView, TensorView)) {
//    return (lhs - rhs, {
//      [lhsShape = lhs.shapeTensor, rhsShape = rhs.shapeTensor] v in
//      return (v.unbroadcast(toShape: lhsShape),
//              -v.unbroadcast(toShape: rhsShape))
//    })
//  }
//
//  @inlinable
//  static func _vjpMultiply(
//    lhs: TensorView, rhs: TensorView
//  ) -> (TensorView, (TensorView) -> (TensorView, TensorView)) {
//    return (lhs * rhs, {
//      [lhsShape = lhs.shapeTensor, rhsShape = rhs.shapeTensor] v in
//      ((rhs * v).unbroadcast(toShape: lhsShape),
//       (lhs * v).unbroadcast(toShape: rhsShape))
//    })
//  }
//
//  @inlinable
//  static func _vjpDivide(
//    lhs: TensorView, rhs: TensorView
//  ) -> (TensorView, (TensorView) -> (TensorView, TensorView)) {
//    return (lhs / rhs, {
//      [lhsShape = lhs.shapeTensor, rhsShape = rhs.shapeTensor] v in
//      ((v / rhs).unbroadcast(toShape: lhsShape),
//       ((-lhs) / rhs.squared() * v).unbroadcast(toShape: rhsShape))
//    })
//  }
//}
//
//extension TensorView where Scalar : TensorFlowFloatingPoint {
//  @inlinable
//  static func _vjpAdd(
//    lhs: TensorView, rhs: Scalar
//  ) -> (TensorView, (TensorView) -> (TensorView, Scalar)) {
//    return (lhs + rhs, { v in (v, v.sum().scalarized()) })
//  }
//
//   @inlinable
//  static func _vjpAdd(
//    lhs: Scalar, rhs: TensorView
//  ) -> (TensorView, (TensorView) -> (Scalar, TensorView)) {
//    return (lhs + rhs, { v in (v.sum().scalarized(), v) })
//  }
//
//  @inlinable
//  static func _vjpSubtract(
//    lhs: TensorView, rhs: Scalar
//  ) -> (TensorView, (TensorView) -> (TensorView, Scalar)) {
//    return (lhs - rhs, { v in (v, 0 - v.sum().scalarized()) })
//  }
//
//  @inlinable
//  static func _vjpSubtract(
//    lhs: Scalar, rhs: TensorView
//  ) -> (TensorView, (TensorView) -> (Scalar, TensorView)) {
//    return (lhs - rhs, { v in (v.sum().scalarized(), 0 - v) })
//  }
//
//  @inlinable
//  static func _vjpMultiply(
//    lhs: TensorView, rhs: Scalar
//  ) -> (TensorView, (TensorView) -> (TensorView, Scalar)) {
//    return (lhs * rhs, { v in (v * rhs, (v * lhs).sum().scalarized()) })
//  }
//
//  @inlinable
//  static func _vjpMultiply(
//    lhs: Scalar, rhs: TensorView
//  ) -> (TensorView, (TensorView) -> (Scalar, TensorView)) {
//    return (lhs * rhs, { v in ((v * rhs).sum().scalarized(), v * lhs) })
//  }
//
//  @inlinable
//  static func _vjpDivide(
//    lhs: TensorView, rhs: Scalar
//  ) -> (TensorView, (TensorView) -> (TensorView, Scalar)) {
//    return (lhs / rhs, { v in
//      (v / rhs, (v * (0 - lhs) / TensorView(rhs).squared()).sum().scalarized())
//    })
//  }
//
//  @inlinable
//  static func _vjpDivide(
//    lhs: Scalar, rhs: TensorView
//  ) -> (TensorView, (TensorView) -> (Scalar, TensorView)) {
//    return (lhs / rhs, { v in
//      ((v / rhs).sum().scalarized(), v * -lhs / rhs.squared())
//    })
//  }
//}
//
//@inlinable
//func _vjpMinMaxHelper<T : TensorFlowFloatingPoint>(
//  _ x: TensorView<T>, _ y: TensorView<T>, originalValue: TensorView<T>, vector: TensorView<T>
//) -> (TensorView<T>, TensorView<T>) {
//  let denom = 1 + TensorView<T>(x .== y)
//  let dfdx = vector * TensorView<T>(x .== originalValue) / denom
//  let dfdy = vector * TensorView<T>(y .== originalValue) / denom
//  return (dfdx.unbroadcast(like: x), dfdy.unbroadcast(like: y))
//}
//
//@inlinable
//func _vjpMax<T : TensorFlowFloatingPoint>(
//  _ x: TensorView<T>, _ y: TensorView<T>
//) -> (TensorView<T>, (TensorView<T>) -> (TensorView<T>, TensorView<T>)) {
//  let value = max(x, y)
//  return (value,
//    { v in _vjpMinMaxHelper(x, y, originalValue: value, vector: v) })
//}
//
//@inlinable
//func _vjpMin<T : TensorFlowFloatingPoint>(
//  _ x: TensorView<T>, _ y: TensorView<T>
//) -> (TensorView<T>, (TensorView<T>) -> (TensorView<T>, TensorView<T>)) {
//  let value = min(x, y)
//  return (value,
//    { v in _vjpMinMaxHelper(x, y, originalValue: value, vector: v) })
//}
//
//@inlinable
//func _vjpPow<T : TensorFlowFloatingPoint>(
//  _ x: TensorView<T>, _ y: TensorView<T>
//) -> (TensorView<T>, (TensorView<T>) -> (TensorView<T>, TensorView<T>)) {
//  let value = pow(x, y)
//  return (value, { v in
//    ((v * y * pow(x, y-1)).unbroadcast(like: x),
//     (v * log(x) * value).unbroadcast(like: y))
//  })
//}
//
////===----------------------------------------------------------------------===//
//// Elementwise unary
////===----------------------------------------------------------------------===//
//
//extension TensorView where Scalar : TensorFlowFloatingPoint {
//  @inlinable
//  static func _vjpNegate(_ x: TensorView) -> (TensorView, (TensorView) -> TensorView) {
//    return (-x, { v in -v })
//  }
//}
//
//@inlinable
//func _vjpLog<T : TensorFlowFloatingPoint>(
//  _ x: TensorView<T>
//) -> (TensorView<T>, (TensorView<T>) -> TensorView<T>) {
//  return (log(x), { v in v / x })
//}
//
//@inlinable
//func _vjpSin<T : TensorFlowFloatingPoint>(
//  _ x: TensorView<T>
//) -> (TensorView<T>, (TensorView<T>) -> TensorView<T>) {
//  return (sin(x), { v in v * cos(x) })
//}
//
//@inlinable
//func _vjpCos<T : TensorFlowFloatingPoint>(
//  _ x: TensorView<T>
//) -> (TensorView<T>, (TensorView<T>) -> TensorView<T>) {
//  return (cos(x), { v in -v * sin(x) })
//}
//
//@inlinable
//func _vjpTan<T : TensorFlowFloatingPoint>(
//  _ x: TensorView<T>
//) -> (TensorView<T>, (TensorView<T>) -> TensorView<T>) {
//  let value = tan(x)
//  return (value, { v in v * (1 + value.squared()) })
//}
//
//@inlinable
//func _vjpSinh<T : TensorFlowFloatingPoint>(
//  _ x: TensorView<T>
//) -> (TensorView<T>, (TensorView<T>) -> TensorView<T>) {
//  return (sinh(x), { v in v * cosh(x) })
//}
//
//@inlinable
//func _vjpCosh<T : TensorFlowFloatingPoint>(
//  _ x: TensorView<T>
//) -> (TensorView<T>, (TensorView<T>) -> TensorView<T>) {
//  return (cosh(x), { v in v * sinh(x) })
//}
//
//@inlinable
//func _vjpTanh<T : TensorFlowFloatingPoint>(
//  _ x: TensorView<T>
//) -> (TensorView<T>, (TensorView<T>) -> TensorView<T>) {
//  let value = tanh(x)
//  return (value, { v in v * (1 - value.squared()) })
//}
//
//@inlinable
//func _vjpExp<T : TensorFlowFloatingPoint>(
//  _ x: TensorView<T>
//) -> (TensorView<T>, (TensorView<T>) -> TensorView<T>) {
//  let value = exp(x)
//  return (value, { v in value * v })
//}
//
//@inlinable
//func _vjpCeil<T : TensorFlowFloatingPoint>(
//  _ x: TensorView<T>
//) -> (TensorView<T>, (TensorView<T>) -> TensorView<T>) {
//  return (ceil(x), { _ in TensorView(0).broadcast(like: x) })
//}
//
//@inlinable
//func _vjpFloor<T : TensorFlowFloatingPoint>(
//  _ x: TensorView<T>
//) -> (TensorView<T>, (TensorView<T>) -> TensorView<T>) {
//  return (floor(x), { _ in TensorView(0).broadcast(like: x) })
//}
//
//@inlinable
//func _vjpSqrt<T : TensorFlowFloatingPoint>(
//  _ x: TensorView<T>
//) -> (TensorView<T>, (TensorView<T>) -> TensorView<T>) {
//  let value = sqrt(x)
//  return (value, { v in v / (2 * value) })
//}
//
//@inlinable
//func _vjpRsqrt<T : TensorFlowFloatingPoint>(
//  _ x: TensorView<T>
//) -> (TensorView<T>, (TensorView<T>) -> TensorView<T>) {
//  let value = rsqrt(x)
//  return (value, { v in -v / 2 * value })
//}
//
//@inlinable
//func _vjpLogSoftmax<T : TensorFlowFloatingPoint>(
//  _ x: TensorView<T>
//) -> (TensorView<T>, (TensorView<T>) -> TensorView<T>) {
//  let value = logSoftmax(x)
//  return (value, { v in
//    v - v.sum(alongAxes: -1) * exp(value)
//  })
//}
//
//extension TensorView where Scalar : TensorFlowFloatingPoint {
//  @inlinable
//  func _vjpSquared() -> (TensorView, (TensorView) -> TensorView) {
//    return (squared(), { 2 * self * $0 })
//  }
//}
//
////===----------------------------------------------------------------------===//
//// Linear algebra
////===----------------------------------------------------------------------===//
//
//@inlinable
//func _vjpMatmul<Scalar : TensorFlowFloatingPoint>(
//  _ lhs: TensorView<Scalar>, _ rhs: TensorView<Scalar>
//) -> (TensorView<Scalar>, (TensorView<Scalar>) -> (TensorView<Scalar>, TensorView<Scalar>)) {
//  let value = matmul(lhs, rhs)
//  return (value, { v in
//    return (matmul(v, rhs.transposed()), matmul(lhs.transposed(), v))
//  })
//}
//
//// TODO: We have to define a custom VJP on • because AD can't yet
//// differentiate generic methods. After AD can differentiate generic methods,
//// remove the custom VJP.
//extension TensorView where Scalar : TensorFlowFloatingPoint {
//  @inlinable
//  static func _vjpMatmulOperator(
//    lhs: TensorView, rhs: TensorView
//  ) -> (TensorView, (TensorView) -> (TensorView, TensorView)) {
//    return _vjpMatmul(lhs, rhs)
//  }
//
//  @inlinable
//  func _vjpTransposed(
//    withPermutations permutations: TensorView<Int32>
//  ) -> (TensorView, (TensorView) -> TensorView) {
//    let value = transposed(withPermutations: permutations)
//    return (value, { $0.transposed(withPermutations: permutations) })
//  }
//
//  @inlinable
//  func _vjpTransposed(
//    withPermutations permutations: [Int32]
//  ) -> (TensorView, (TensorView) -> TensorView) {
//    let value = transposed(withPermutations: permutations)
//    return (value, { $0.transposed(withPermutations: permutations) })
//  }
//
//  @inlinable
//  func _vjpTransposed(
//    withPermutations permutations: Int32...
//  ) -> (TensorView, (TensorView) -> TensorView) {
//    let value = transposed(withPermutations: permutations)
//    return (value, { $0.transposed(withPermutations: permutations) })
//  }
//
//  @inlinable
//  func _vjpTransposed() -> (TensorView, (TensorView) -> TensorView) {
//    return (transposed(), { $0.transposed() })
//  }
//}
//
////===----------------------------------------------------------------------===//
//// Shape transformations
////===----------------------------------------------------------------------===//
//
//extension TensorView where Scalar : TensorFlowFloatingPoint {
//  @inlinable
//  func _vjpReshaped(
//    toShape newShape: TensorView<Int32>
//  ) -> (TensorView, (TensorView) -> TensorView) {
//    let value = reshaped(toShape: newShape)
//    return (value, { v in
//      return v.reshaped(toShape: self.shapeTensor)
//    })
//  }
//
//  @inlinable
//  func _vjpExpandingShape(
//    at shapeIndex: Int32
//  ) -> (TensorView, (TensorView) -> TensorView) {
//    let value = expandingShape(at: shapeIndex)
//    return (value, { v in
//      return v.squeezingShape(at: shapeIndex)
//    })
//  }
//}
//
////===----------------------------------------------------------------------===//
//// Reduction
////===----------------------------------------------------------------------===//
//
//extension TensorView where Scalar : TensorFlowFloatingPoint {
//  @inlinable
//  func _vjpMean() -> (TensorView, (TensorView) -> TensorView) {
//    return (mean(), { [shape = shapeTensor, count = scalarCountTensor] in
//      $0.broadcast(toShape: shape) / TensorView(count)
//    })
//  }
//
//  @inlinable
//  func _vjpSum() -> (TensorView, (TensorView) -> TensorView) {
//    return (sum(), { [shape = shapeTensor] in $0.broadcast(toShape: shape) })
//  }
//
//  @inlinable
//  func _vjpMean(alongAxes axes: [Int32]) -> (TensorView, (TensorView) -> TensorView) {
//    let value = mean(alongAxes: axes)
//    return (value, { [shape = shapeTensor, count = scalarCountTensor] in
//      $0.broadcast(toShape: shape) / TensorView(count)
//    })
//  }
//
//  @inlinable
//  func _vjpSum(alongAxes axes: [Int32]) -> (TensorView, (TensorView) -> TensorView) {
//    let value = sum(alongAxes: axes)
//    return (value, { [shape = shapeTensor] in $0.broadcast(toShape: shape) })
//  }
//}
//
////===----------------------------------------------------------------------===//
//// Normalization
////===----------------------------------------------------------------------===//
//
//extension TensorView where Scalar : TensorFlowFloatingPoint {
//  // TODO: Verify that these calculations are correct.
//  @inlinable
//  func _vjpBatchNormalized(
//    alongAxis axis: Int32,
//    offset: TensorView,
//    scale: TensorView,
//    epsilon: Scalar
//  ) -> (TensorView, (TensorView) -> (TensorView, TensorView, TensorView)) {
//    let value = batchNormalized(alongAxis: axis, offset: offset, scale: scale,
//                                epsilon: epsilon)
//    return (value, { v in
//      let mean = self.mean(alongAxes: axis)
//      let squaredDiff: TensorView = Raw.squaredDifference(self, mean)
//      let variance = squaredDiff.mean(alongAxes: axis)
//
//      let diff = self - mean
//      let inv = rsqrt(variance + epsilon)
//      let norm = diff * inv
//
//      let dNorm = v * scale
//      let dVariance = -(dNorm * diff).sum(alongAxes: axis) / 2 * pow(inv, -3)
//      let dMean = (-dNorm * inv).sum(alongAxes: axis) +
//        dVariance * (-diff * 2).mean(alongAxes: axis)
//      let dOffset = v.sum(alongAxes: axis)
//      let dScale = (norm * v).sum(alongAxes: axis)
//      let dim = TensorView(TensorView<Int32>(self.shapeTensor[axis]))
//      let tmp = (dNorm * inv) + (dVariance * 2 * dMean / dim)
//      let dSelf = tmp + (dMean / dim)
//      return (dSelf, dOffset, dScale)
//    })
//  }
//}
//
////===----------------------------------------------------------------------===//
//// Convolution and pooling
////===----------------------------------------------------------------------===//
//
//extension TensorView where Scalar : TensorFlowFloatingPoint {
//  /// TensorFlow builtin conv2d gradient helper for the input.
//  @inlinable
//  @differentiable(
//    wrt: (filter, backpropOutput),
//    vjp: _vjpTFConv2DBackpropInput(_:_:_:_:_:)
//  )
//  func _TFConv2DBackpropInput(
//    shape: TensorView<Int32>,
//    filter: TensorView,
//    backpropOutput: TensorView,
//    strides: (Int32, Int32, Int32, Int32),
//    padding: Padding
//  ) -> TensorView {
//    return Raw.conv2DBackpropInput(
//      inputSizes: shape,
//      filter: filter,
//      outBackprop: backpropOutput,
//      strides: [strides.0, strides.1, strides.2, strides.3],
//      padding: padding.raw)
//  }
//
//  /// TensorFlow builtin conv2d gradient helper for the filter.
//  @inlinable
//  @differentiable(
//    wrt: (input, backpropOutput),
//    vjp: _vjpTFConv2DBackpropFilter(_:_:_:_:_:)
//  )
//  func _TFConv2DBackpropFilter(
//    input: TensorView,
//    filterSizes: TensorView<Int32>,
//    backpropOutput: TensorView,
//    strides: (Int32, Int32, Int32, Int32),
//    padding: Padding
//  ) -> TensorView {
//    return Raw.conv2DBackpropFilter(
//      input,
//      filterSizes: filterSizes,
//      outBackprop: backpropOutput,
//      strides: [strides.0, strides.1, strides.2, strides.3],
//      padding: padding.raw)
//  }
//
//  @inlinable
//  func _vjpTFConv2DBackpropInput(
//    _ shape: TensorView<Int32>,
//    _ filter: TensorView,
//    _ backpropOutput: TensorView,
//    _ strides: (Int32, Int32, Int32, Int32),
//    _ padding: Padding
//  ) -> (TensorView, (TensorView) -> (TensorView, TensorView)) {
//    let value = _TFConv2DBackpropInput(shape: shape, filter: filter,
//                                       backpropOutput: backpropOutput,
//                                       strides: strides, padding: padding)
//    return (value, { v in
//      return (
//        self._TFConv2DBackpropFilter(input: v, filterSizes: shape,
//                                     backpropOutput: backpropOutput,
//                                     strides: strides, padding: padding),
//        v.convolved2D(withFilter: filter, strides: strides, padding: padding)
//      )
//    })
//  }
//
//  @inlinable
//  func _vjpTFConv2DBackpropFilter(
//    _ input: TensorView,
//    _ filterSizes: TensorView<Int32>,
//    _ backpropOutput: TensorView,
//    _ strides: (Int32, Int32, Int32, Int32),
//    _ padding: Padding
//  ) -> (TensorView, (TensorView) -> (TensorView, TensorView)) {
//    let value = _TFConv2DBackpropFilter(input: input, filterSizes: filterSizes,
//                                        backpropOutput: backpropOutput,
//                                        strides: strides, padding: padding)
//    return (value, { v in
//      return (
//        self._TFConv2DBackpropInput(shape: filterSizes, filter: v,
//                                    backpropOutput: backpropOutput,
//                                    strides: strides, padding: padding),
//        input.convolved2D(withFilter: v, strides: strides, padding: padding)
//      )
//    })
//  }
//
//  @inlinable
//  func _vjpConvolved2D(
//    filter: TensorView,
//    strides: (Int32, Int32, Int32, Int32),
//    padding: Padding
//  ) -> (TensorView, (TensorView) -> (TensorView, TensorView)) {
//    let value = convolved2D(withFilter: filter, strides: strides,
//                            padding: padding)
//    return (value, { v in
//      return (
//        self._TFConv2DBackpropInput(
//          shape: self.shapeTensor, filter: filter, backpropOutput: v,
//          strides: strides, padding: padding
//        ),
//        self._TFConv2DBackpropFilter(
//          input: self, filterSizes: filter.shapeTensor, backpropOutput: v,
//          strides: strides, padding: padding
//        )
//      )
//    })
//  }
//
//  @inlinable
//  func _vjpMaxPooled(
//    kernelSize: (Int32, Int32, Int32, Int32),
//    strides: (Int32, Int32, Int32, Int32),
//    padding: Padding
//  ) -> (TensorView, (TensorView) -> TensorView) {
//    // TODO: Currently this is not higher order differentiable. Redefine in
//    // closed form.
//    let value = maxPooled(kernelSize: kernelSize, strides: strides,
//                          padding: padding)
//    return (value, { v in
//      return Raw.maxPoolGradV2(
//        origInput: self,
//        origOutput: value,
//        grad: v,
//        ksize: TensorView<Int32>(kernelSize),
//        strides: TensorView<Int32>(strides),
//        padding: padding.raw
//      )
//    })
//  }
//
//  @inlinable
//  func _vjpAveragePooled(
//    kernelSize: (Int32, Int32, Int32, Int32),
//    strides: (Int32, Int32, Int32, Int32),
//    padding: Padding
//  ) -> (TensorView, (TensorView) -> TensorView) {
//    // TODO: Currently this is not higher order differentiable. Redefine in
//    // closed form.
//    let value = averagePooled(kernelSize: kernelSize, strides: strides,
//                              padding: padding)
//    return (value, { v in
//      return Raw.avgPoolGrad(
//        origInputShape: self.shapeTensor,
//        grad: v,
//        ksize: [kernelSize.0, kernelSize.1, kernelSize.2, kernelSize.3],
//        strides: [strides.0, strides.1, strides.2, strides.3],
//        padding: padding.raw
//      )
//    })
//  }
//}
//
////===----------------------------------------------------------------------===//
//// Composite math
////===----------------------------------------------------------------------===//
//
//@inlinable
//func _vjpSigmoid<T : TensorFlowFloatingPoint>(
//  _ x: TensorView<T>
//) -> (TensorView<T>, (TensorView<T>) -> TensorView<T>) {
//  let value = sigmoid(x)
//  return (value, { v in Raw.sigmoidGrad(value, dy: v) })
//}
//
//@inlinable
//func _vjpSoftmax<T : TensorFlowFloatingPoint>(
//  _ x: TensorView<T>
//) -> (TensorView<T>, (TensorView<T>) -> TensorView<T>) {
//  let value = softmax(x)
//  return (value, { v in
//    let sumChannels = (v * value).sum(alongAxes: -1)
//    return (v - sumChannels) * value
//  })
//}
//
//@inlinable
//func _vjpRelu<T : TensorFlowFloatingPoint>(
//  _ x: TensorView<T>
//) -> (TensorView<T>, (TensorView<T>) -> TensorView<T>) {
//  return (relu(x), { v in TensorView(x .> 0) * v })
//}
