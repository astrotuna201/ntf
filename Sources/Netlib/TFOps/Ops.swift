//******************************************************************************
// Created by Edward Connell on 3/24/19
// Copyright © 2019 Connell Research. All rights reserved.
//
// This file API implementation is derived from the Swift 4 TensorFlow API
// which cites the following Copyright
//
// Copyright (c) 2014 - 2017 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// This file contains definitions of most TensorView operations, however the
// underlying implementation is based on Netlib and not on TensorFlow
//
import Foundation
import TensorFlow

infix operator ++  : AdditionPrecedence
infix operator .<  : ComparisonPrecedence
infix operator .<= : ComparisonPrecedence
infix operator .>= : ComparisonPrecedence
infix operator .>  : ComparisonPrecedence
infix operator .== : ComparisonPrecedence
infix operator .!= : ComparisonPrecedence
infix operator .=

// TODO:
// - Consider explicit broadcasting for elementwise binary ops when
//   scalarization and rank getter are implemented.

// Additive group
public extension TensorDataView where Self.Scalar: Numeric & AnyNumeric {
    //==========================================================================
    /// Adds two tensors and produces their sum.
    /// - Note: `+` supports broadcasting.
    // operator
    @inlinable @inline(__always)
    //  @differentiable(vjp: _vjpAdd(lhs:rhs:) where Scalar : TensorFlowFloatingPoint)
    static func + (lhs: Self, rhs: Self) -> Self {
        return _ThreadLocal.value.catchError {
            return try add(lhs, rhs)
        }
    }

    // operator (Self + scalar)
    @inlinable @inline(__always)
    //  @differentiable(vjp: _vjpAdd(lhs:rhs:) where Scalar : TensorFlowFloatingPoint)
    static func +<S: AnyNumeric>(lhs: Self, rhs: S) -> Self {
        let scalarValue = ScalarTensor(value: Scalar(any: rhs))
        return _ThreadLocal.value.catchError {
            return try add(lhs, scalarValue)
        }
    }

    //==========================================================================
    /// Subtracts one TensorView from another and produces their difference.
    /// - Note: `-` supports broadcasting.
    @inlinable @inline(__always)
    //    @differentiable(vjp: _vjpSubtract(lhs:rhs:) where Scalar: TensorFlowFloatingPoint)
    static func subtract(_ lhs: Self, _ rhs: Self, result: inout Self,
                         using deviceStream: DeviceStream? = nil) throws {

        let stream = deviceStream ?? _ThreadLocal.value.defaultStream
        try stream.subtract(lhs: lhs, rhs: rhs, result: &result)
    }

    // returns new view
    @inlinable @inline(__always)
    //    @differentiable(vjp: _vjpSubtract(lhs:rhs:) where Scalar: TensorFlowFloatingPoint)
    static func subtract(_ lhs: Self, _ rhs: Self,
                         using stream: DeviceStream? = nil) throws -> Self {

        var result = Self.init(denseLike: lhs)
        try subtract(lhs, rhs, result: &result, using: stream)
        return result
    }
    
    // operator
    @inlinable @inline(__always)
//    @differentiable(vjp: _vjpSubtract(lhs:rhs:) where Scalar: TensorFlowFloatingPoint)
    static func - (lhs: Self, rhs: Self) -> Self {
        return _ThreadLocal.value.catchError {
            return try subtract(lhs, rhs)
        }
    }
}

//==============================================================================
/// Adds two tensors and produces their sum.
/// - Note: `+` supports broadcasting.
@inlinable @inline(__always)
//  @differentiable(vjp: _vjpAdd(lhs:rhs:) where Scalar : TensorFlowFloatingPoint)
public func add<TL, TR>(_ lhs: TL, _ rhs: TR, result: inout TL,
                using deviceStream: DeviceStream? = nil) throws
    where
    TL: TensorDataView, TL.Scalar: Numeric,
    TR: TensorDataView, TR.Scalar: Numeric
{
    let stream = deviceStream ?? _ThreadLocal.value.defaultStream
    try stream.add(lhs: lhs, rhs: rhs, result: &result)
}

// returns new view
@inlinable @inline(__always)
//  @differentiable(vjp: _vjpAdd(lhs:rhs:) where Scalar : TensorFlowFloatingPoint)
public func add<TL, TR>(_ lhs: TL, _ rhs: TR,
                        using deviceStream: DeviceStream? = nil) throws -> TL
    where
    TL: TensorDataView, TL.Scalar: Numeric,
    TR: TensorDataView, TR.Scalar: Numeric
{
    var result = TL.init(denseLike: lhs)
    try add(lhs, rhs, result: &result, using: deviceStream)
    return result
}


////==============================================================================
//// Vector space
//extension TensorView: VectorNumeric where Scalar: Numeric {
//    /// Multiplies the scalar with every scalar of the TensorView and produces the
//    /// product.
//    @inlinable @inline(__always)
//    //  @differentiable(
//    //    vjp: _vjpMultiply(lhs:rhs:)
//    //    where Scalar : TensorFlowFloatingPoint
//    //  )
//    public static func * (lhs: Scalar, rhs: TensorView) -> TensorView {
//        return TensorView(lhs) * rhs
//    }
//}
//
//extension TensorView : ShapedVectorNumeric where Scalar : Numeric { }
//
//extension TensorView: Differentiable where Scalar: TensorFlowFloatingPoint {
//    public typealias TangentVector = TensorView
//    public typealias CotangentVector = TensorView
//    public typealias AllDifferentiableVariables = TensorView
//    @inlinable @inline(__always)
//    public func tangentVector(from cotangent: CotangentVector) -> TangentVector {
//        return cotangent
//    }
//}
//
////==============================================================================
//// Additional element-wise operators
//public extension TensorView where Scalar: Numeric {
//    /// Adds the scalar to every scalar of the TensorView and produces the sum.
//    @inlinable @inline(__always)
//    //  @differentiable(vjp: _vjpAdd(lhs:rhs:) where Scalar : TensorFlowFloatingPoint)
//    static func + (lhs: Scalar, rhs: TensorView) -> TensorView {
//        return TensorView(lhs) + rhs
//    }
//
//    /// Adds the scalar to every scalar of the TensorView and produces the sum.
//    @inlinable @inline(__always)
//    //  @differentiable(vjp: _vjpAdd(lhs:rhs:) where Scalar : TensorFlowFloatingPoint)
//    static func + (lhs: TensorView, rhs: Scalar) -> TensorView {
//        return lhs + TensorView(rhs)
//    }
//
//    /// Subtracts the scalar from every scalar of the TensorView and produces the
//    /// difference.
//    @inlinable @inline(__always)
//    //  @differentiable(
//    //    vjp: _vjpSubtract(lhs:rhs:)
//    //    where Scalar : TensorFlowFloatingPoint
//    //  )
//    static func - (lhs: Scalar, rhs: TensorView) -> TensorView {
//        return TensorView(lhs) - rhs
//    }
//
//    /// Subtracts the scalar from every scalar of the TensorView and produces the
//    /// difference.
//    @inlinable @inline(__always)
//    //  @differentiable(
//    //    vjp: _vjpSubtract(lhs:rhs:)
//    //    where Scalar : TensorFlowFloatingPoint
//    //  )
//    static func - (lhs: TensorView, rhs: Scalar) -> TensorView {
//        return lhs - TensorView(rhs)
//    }
//
//    /// Adds two tensors and stores the result in the left-hand-side variable.
//    /// - Note: `+=` supports broadcasting.
//    @inlinable @inline(__always)
//    static func += (lhs: inout TensorView, rhs: TensorView) {
//        lhs = lhs + rhs
//    }
//
//    /// Adds the scalar to every scalar of the TensorView and stores the result in the
//    /// left-hand-side variable.
//    @inlinable @inline(__always)
//    static func += (lhs: inout TensorView, rhs: Scalar) {
//        lhs = lhs + rhs
//    }
//
//    /// Subtracts the second TensorView from the first and stores the result in the
//    /// left-hand-side variable.
//    /// - Note: `-=` supports broadcasting.
//    @inlinable @inline(__always)
//    static func -= (lhs: inout TensorView, rhs: TensorView) {
//        lhs = lhs - rhs
//    }
//
//    /// Subtracts the scalar from every scalar of the TensorView and stores the result
//    /// in the left-hand-side variable.
//    @inlinable @inline(__always)
//    static func -= (lhs: inout TensorView, rhs: Scalar) {
//        lhs = lhs - rhs
//    }
//
//    //--------------------------------------------------------------------------
//    /// Multiplies two tensors and produces their product.
//    /// - Note: `*` supports broadcasting.
//    @inlinable @inline(__always)
//    //  @differentiable(
//    //    vjp: _vjpMultiply(lhs:rhs:)
//    //    where Scalar : TensorFlowFloatingPoint
//    //  )
//    static func mul(_ lhs: TensorView, _ rhs: TensorView,
//                    result: inout TensorView,
//                    using deviceStream: DeviceStream? = nil) throws {
//        assert(lhs.shape == rhs.shape)
//        let stream = deviceStream ?? _ThreadLocal.value.defaultStream
//        try stream.mul(lhs: lhs, rhs: rhs, result: &result)
//    }
//
//    // returns new view
//    @inlinable @inline(__always)
//    //  @differentiable(
//    //    vjp: _vjpMultiply(lhs:rhs:)
//    //    where Scalar : TensorFlowFloatingPoint
//    //  )
//    static func mul(_ lhs: TensorView, _ rhs: TensorView,
//                    using deviceStream: DeviceStream? = nil) throws -> TensorView {
//        var result = TensorView(shape: lhs.shape)
//        try add(lhs, rhs, result: &result, using: deviceStream)
//        return result
//    }
//
//    // operator
//    @inlinable @inline(__always)
//    //  @differentiable(
//    //    vjp: _vjpMultiply(lhs:rhs:)
//    //    where Scalar : TensorFlowFloatingPoint
//    //  )
//    static func * (lhs: TensorView, rhs: TensorView) -> TensorView {
//        return _ThreadLocal.value.catchError {
//            return try mul(lhs, rhs)
//        }
//    }
//
//    //--------------------------------------------------------------------------
//    /// Multiplies the scalar with every scalar of the TensorView and produces the
//    /// product.
//    @inlinable @inline(__always)
//    //  @differentiable(
//    //    vjp: _vjpMultiply(lhs:rhs:)
//    //    where Scalar : TensorFlowFloatingPoint
//    //  )
//    static func * (lhs: TensorView, rhs: Scalar) -> TensorView {
//        return lhs * TensorView(rhs)
//    }
//
//    /// Multiplies two tensors and stores the result in the left-hand-side
//    /// variable.
//    /// - Note: `*=` supports broadcasting.
//    @inlinable @inline(__always)
//    static func *= (lhs: inout TensorView, rhs: TensorView) {
//        lhs = lhs * rhs
//    }
//
//    @inlinable @inline(__always)
//    static func *= (lhs: inout TensorView, rhs: Scalar) {
//        lhs = lhs * rhs
//    }
//
//    //--------------------------------------------------------------------------
//    /// Returns the quotient of dividing the first TensorView by the second.
//    /// - Note: `/` supports broadcasting.
//    @inlinable @inline(__always)
//    //  @differentiable(
//    //    vjp: _vjpDivide(lhs:rhs:)
//    //    where Scalar : TensorFlowFloatingPoint
//    //  )
//    static func div(_ lhs: TensorView, _ rhs: TensorView,
//                    result: inout TensorView,
//                    using deviceStream: DeviceStream? = nil) throws {
//        assert(lhs.shape == rhs.shape)
//        let stream = deviceStream ?? _ThreadLocal.value.defaultStream
//        try stream.div(lhs: lhs, rhs: rhs, result: &result)
//    }
//
//    // returns new view
//    @inlinable @inline(__always)
//    //  @differentiable(
//    //    vjp: _vjpDivide(lhs:rhs:)
//    //    where Scalar : TensorFlowFloatingPoint
//    //  )
//    static func div(_ lhs: TensorView, _ rhs: TensorView,
//                    using deviceStream: DeviceStream? = nil) throws -> TensorView {
//        var result = TensorView(shape: lhs.shape)
//        try div(lhs, rhs, result: &result, using: deviceStream)
//        return result
//    }
//
//    // operator
//    @inlinable @inline(__always)
//    //  @differentiable(
//    //    vjp: _vjpDivide(lhs:rhs:)
//    //    where Scalar : TensorFlowFloatingPoint
//    //  )
//    static func / (lhs: TensorView, rhs: TensorView) -> TensorView {
//        return _ThreadLocal.value.catchError {
//            return try div(lhs, rhs)
//        }
//    }
//
//    //--------------------------------------------------------------------------
//    /// Returns the quotient of dividing the scalar by the TensorView, broadcasting
//    /// the scalar.
//    @inlinable @inline(__always)
//    //  @differentiable(
//    //    vjp: _vjpDivide(lhs:rhs:)
//    //    where Scalar : TensorFlowFloatingPoint
//    //  )
//    static func / (lhs: Scalar, rhs: TensorView) -> TensorView {
//        return TensorView(lhs) / rhs
//    }
//
//    /// Returns the quotient of dividing the TensorView by the scalar, broadcasting
//    /// the scalar.
//    @inlinable @inline(__always)
//    //  @differentiable(
//    //    vjp: _vjpDivide(lhs:rhs:)
//    //    where Scalar : TensorFlowFloatingPoint
//    //  )
//    static func / (lhs: TensorView, rhs: Scalar) -> TensorView {
//        return lhs / TensorView(rhs)
//    }
//
//    /// Divides the first TensorView by the second and stores the quotient in the
//    /// left-hand-side variable.
//    @inlinable @inline(__always)
//    static func /= (lhs: inout TensorView, rhs: TensorView) {
//        lhs = lhs / rhs
//    }
//
//    /// Divides the TensorView by the scalar, broadcasting the scalar, and stores the
//    /// quotient in the left-hand-side variable.
//    @inlinable @inline(__always)
//    static func /= (lhs: inout TensorView, rhs: Scalar) {
//        lhs = lhs / rhs
//    }
//
//    //--------------------------------------------------------------------------
//    /// Returns the remainder of dividing the first TensorView by the second.
//    /// - Note: `%` supports broadcasting.
//    @inlinable @inline(__always)
//    static func mod(_ lhs: TensorView, _ rhs: TensorView,
//                    result: inout TensorView,
//                    using deviceStream: DeviceStream? = nil) throws {
//        assert(lhs.shape == rhs.shape)
//        let stream = deviceStream ?? _ThreadLocal.value.defaultStream
//        try stream.mod(lhs: lhs, rhs: rhs, result: &result)
//    }
//
//    // returns new view
//    @inlinable @inline(__always)
//    static func mod(_ lhs: TensorView, _ rhs: TensorView,
//                    using deviceStream: DeviceStream? = nil) throws -> TensorView {
//        var result = TensorView(shape: lhs.shape)
//        try mod(lhs, rhs, result: &result, using: deviceStream)
//        return result
//    }
//
//    // operator
//    @inlinable @inline(__always)
//    static func % (lhs: TensorView, rhs: TensorView) -> TensorView {
//        return _ThreadLocal.value.catchError {
//            return try mod(lhs, rhs)
//        }
//    }
//
//    //--------------------------------------------------------------------------
//    /// Returns the remainder of dividing the TensorView by the scalar, broadcasting
//    /// the scalar.
//    @inlinable @inline(__always)
//    static func % (lhs: TensorView, rhs: Scalar) -> TensorView {
//        return lhs % TensorView(rhs)
//    }
//
//    /// Returns the remainder of dividing the scalar by the TensorView, broadcasting
//    /// the scalar.
//    @inlinable @inline(__always)
//    static func % (lhs: Scalar, rhs: TensorView) -> TensorView {
//        return TensorView(lhs) % rhs
//    }
//
//    /// Divides the first TensorView by the second and stores the remainder in the
//    /// left-hand-side variable.
//    @inlinable @inline(__always)
//    static func %= (lhs: inout TensorView, rhs: TensorView) {
//        lhs = lhs % rhs
//    }
//
//    /// Divides the TensorView by the scalar and stores the remainder in the
//    /// left-hand-side variable.
//    @inlinable @inline(__always)
//    static func %= (lhs: inout TensorView, rhs: Scalar) {
//        lhs = lhs % rhs
//    }
//}
//
////==============================================================================
//// Linear algebra
//
///// Performs matrix multiplication with another TensorView and produces the
///// result.
//@inlinable @inline(__always)
////@differentiable(
////  vjp: _vjpMatmul(_:_:)
////  where Scalar : TensorFlowFloatingPoint
////)
//public func matmul<Scalar: Numeric>(
//    _ lhs: TensorView<Scalar>, _ rhs: TensorView<Scalar>,
//    result: inout TensorView<Scalar>,
//    using deviceStream: DeviceStream? = nil) throws {
//
//    assert(lhs.shape == rhs.shape)
//    let stream = deviceStream ?? _ThreadLocal.value.defaultStream
//    try stream.matmul(lhs: lhs, rhs: rhs, result: &result)
//}
//
//// returns new view
//@inlinable @inline(__always)
////@differentiable(
////  vjp: _vjpMatmul(_:_:)
////  where Scalar : TensorFlowFloatingPoint
////)
//public func matmul<Scalar: Numeric>(
//    _ lhs: TensorView<Scalar>, _ rhs: TensorView<Scalar>,
//    using deviceStream: DeviceStream? = nil) throws -> TensorView<Scalar> {
//
//    var result = TensorView<Scalar>(shape: lhs.shape)
//    try matmul(lhs, rhs, result: &result, using: deviceStream)
//    return result
//}
//
//infix operator • : MultiplicationPrecedence
//
//public extension TensorView where Scalar : Numeric {
//    // TODO: We have to define a custom VJP on • because AD can't yet
//    // differentiate generic methods. After AD can differentiate generic methods,
//    // remove the custom VJP.
//
//    /// Performs matrix multiplication between two tensors and produces the
//    /// result.
//    @inlinable @inline(__always)
//    //  @differentiable(
//    //    vjp: _vjpMatmulOperator(lhs:rhs:)
//    //    where Scalar : TensorFlowFloatingPoint
//    //  )
//    static func • (lhs: TensorView, rhs: TensorView) throws -> TensorView {
//        return try matmul(lhs, rhs)
//    }
//}
//
////==============================================================================
//// Element-wise binary comparison
//public extension TensorView where Scalar : Numeric & Comparable {
//    //--------------------------------------------------------------------------
//    /// Computes `lhs < rhs` element-wise and returns a `TensorView` of Boolean
//    /// scalars.
//    @inlinable @inline(__always)
//    static func less(_ lhs: TensorView, _ rhs: TensorView,
//                    result: inout TensorView<Bool>,
//                    using deviceStream: DeviceStream? = nil) throws {
//        assert(lhs.shape == rhs.shape)
//        let stream = deviceStream ?? _ThreadLocal.value.defaultStream
//        try stream.less(lhs: lhs, rhs: rhs, result: &result)
//    }
//
//    // returns new view
//    @inlinable @inline(__always)
//    static func less(
//        _ lhs: TensorView, _ rhs: TensorView,
//        using deviceStream: DeviceStream? = nil) throws -> TensorView<Bool> {
//
//        var result = TensorView<Bool>(shape: lhs.shape)
//        try less(lhs, rhs, result: &result, using: deviceStream)
//        return result
//    }
//
//    // operator
//    @inlinable @inline(__always)
//    static func .< (lhs: TensorView, rhs: TensorView) -> TensorView<Bool> {
//        return _ThreadLocal.value.catchError {
//            return try less(lhs, rhs)
//        }
//    }
//
//
//
//
//    /// Computes `lhs <= rhs` element-wise and returns a `TensorView` of Boolean
//    /// scalars.
//    @inlinable @inline(__always)
//    static func .<= (lhs: TensorView, rhs: TensorView) -> TensorView<Bool> {
//        fatalError("Not implemented")
//        // FunctionId.LessEqual
//        //    return Raw.lessEqual(lhs, rhs)
//    }
//
//    /// Computes `lhs > rhs` element-wise and returns a `TensorView` of Boolean
//    /// scalars.
//    @inlinable @inline(__always)
//    static func .> (lhs: TensorView, rhs: TensorView) -> TensorView<Bool> {
//        fatalError("Not implemented")
//        // FunctionId.Greater
//        //    return Raw.greater(lhs, rhs)
//    }
//
//    /// Computes `lhs >= rhs` element-wise and returns a `TensorView` of Boolean
//    /// scalars.
//    @inlinable @inline(__always)
//    static func .>= (lhs: TensorView, rhs: TensorView) -> TensorView<Bool> {
//        fatalError("Not implemented")
//        // FunctionId.GreaterEqual
//        //    return Raw.greaterEqual(lhs, rhs)
//    }
//
//    /// Computes `lhs < rhs` element-wise and returns a `TensorView` of Boolean
//    /// scalars.
//    /// - Note: `.<` supports broadcasting.
//    @inlinable @inline(__always)
//    static func .< (lhs: Scalar, rhs: TensorView) -> TensorView<Bool> {
//        fatalError("Not implemented")
//        // FunctionId.Less
//        //    return Raw.less(TensorView(lhs), rhs)
//    }
//
//    /// Computes `lhs <= rhs` element-wise and returns a `TensorView` of Boolean
//    /// scalars.
//    /// - Note: `.<=` supports broadcasting.
//    @inlinable @inline(__always)
//    static func .<= (lhs: Scalar, rhs: TensorView) -> TensorView<Bool> {
//        fatalError("Not implemented")
//        // FunctionId.LessEqual
//        //    return Raw.lessEqual(TensorView(lhs), rhs)
//    }
//
//    /// Computes `lhs > rhs` element-wise and returns a `TensorView` of Boolean
//    /// scalars.
//    /// - Note: `.>` supports broadcasting.
//    @inlinable @inline(__always)
//    static func .> (lhs: Scalar, rhs: TensorView) -> TensorView<Bool> {
//        fatalError("Not implemented")
//        // FunctionId.Greater
//        //    return Raw.greater(TensorView(lhs), rhs)
//    }
//
//    /// Computes `lhs >= rhs` element-wise and returns a `TensorView` of Boolean
//    /// scalars.
//    /// - Note: `.>=` supports broadcasting.
//    @inlinable @inline(__always)
//    static func .>= (lhs: Scalar, rhs: TensorView) -> TensorView<Bool> {
//        fatalError("Not implemented")
//        // FunctionId.GreaterEqual
//        //    return Raw.greaterEqual(TensorView(lhs), rhs)
//    }
//
//    /// Computes `lhs < rhs` element-wise and returns a `TensorView` of Boolean
//    /// scalars.
//    /// - Note: `.<` supports broadcasting.
//    @inlinable @inline(__always)
//    static func .< (lhs: TensorView, rhs: Scalar) -> TensorView<Bool> {
//        fatalError("Not implemented")
//        // FunctionId.Less
//        //    return Raw.less(lhs, TensorView(rhs))
//    }
//
//    /// Computes `lhs <= rhs` element-wise and returns a `TensorView` of Boolean
//    /// scalars.
//    /// - Note: `.<=` supports broadcasting.
//    @inlinable @inline(__always)
//    static func .<= (lhs: TensorView, rhs: Scalar) -> TensorView<Bool> {
//        fatalError("Not implemented")
//        // FunctionId.LessEqual
//        //    return Raw.lessEqual(lhs, TensorView(rhs))
//    }
//
//    /// Computes `lhs > rhs` element-wise and returns a `TensorView` of Boolean
//    /// scalars.
//    /// - Note: `.>` supports broadcasting.
//    @inlinable @inline(__always)
//    static func .> (lhs: TensorView, rhs: Scalar) -> TensorView<Bool> {
//        fatalError("Not implemented")
//        // FunctionId.Greater
//        //    return Raw.greater(lhs, TensorView(rhs))
//    }
//
//    /// Computes `lhs >= rhs` element-wise and returns a `TensorView` of Boolean
//    /// scalars.
//    /// - Note: `.>=` supports broadcasting.
//    @inlinable @inline(__always)
//    static func .>= (lhs: TensorView, rhs: Scalar) -> TensorView<Bool> {
//        fatalError("Not implemented")
//        // FunctionId.GreaterEqual
//        //    return Raw.greaterEqual(lhs, TensorView(rhs))
//    }
//}
//
//extension TensorView : Comparable where Scalar : Numeric & Comparable {
//    /// Returns a Boolean value indicating whether the value of the first argument
//    /// is lexicographically less than that of the second argument.
//    @inlinable @inline(__always)
//    public static func < (lhs: TensorView, rhs: TensorView) -> Bool {
//        return (lhs .< rhs).all()
//    }
//
//    /// Returns a Boolean value indicating whether the value of the first argument
//    /// is lexicographically less than or equal to that of the second argument.
//    @inlinable @inline(__always)
//    public static func <= (lhs: TensorView, rhs: TensorView) -> Bool {
//        return (lhs .<= rhs).all()
//    }
//
//    /// Returns a Boolean value indicating whether the value of the first argument
//    /// is lexicographically greater than that of the second argument.
//    @inlinable @inline(__always)
//    public static func > (lhs: TensorView, rhs: TensorView) -> Bool {
//        return (lhs .> rhs).all()
//    }
//
//    /// Returns a Boolean value indicating whether the value of the first argument
//    /// is lexicographically greater than or equal to that of the second argument.
//    @inlinable @inline(__always)
//    public static func >= (lhs: TensorView, rhs: TensorView) -> Bool {
//        return (lhs .>= rhs).all()
//    }
//}
//
//public extension TensorView where Scalar : Numeric & Comparable {
//    /// Returns a Boolean value indicating whether the value of the first argument
//    /// is lexicographically less than that of the second argument.
//    @inlinable @inline(__always)
//    static func < (lhs: TensorView, rhs: Scalar) -> Bool {
//        return (lhs .< rhs).all()
//    }
//
//    /// Returns a Boolean value indicating whether the value of the first argument
//    /// is lexicographically less than or equal to that of the second argument.
//    @inlinable @inline(__always)
//    static func <= (lhs: TensorView, rhs: Scalar) -> Bool {
//        return (lhs .<= rhs).all()
//    }
//
//    /// Returns a Boolean value indicating whether the value of the first argument
//    /// is lexicographically greater than that of the second argument.
//    @inlinable @inline(__always)
//    static func > (lhs: TensorView, rhs: Scalar) -> Bool {
//        return (lhs .> rhs).all()
//    }
//
//    /// Returns a Boolean value indicating whether the value of the first argument
//    /// is lexicographically greater than or equal to that of the second argument.
//    @inlinable @inline(__always)
//    static func >= (lhs: TensorView, rhs: Scalar) -> Bool {
//        return (lhs .>= rhs).all()
//    }
//}
//
//public extension TensorView where Scalar : Equatable {
//    /// Computes `lhs != rhs` element-wise and returns a `TensorView` of Boolean
//    /// scalars.
//    /// - Note: `.==` supports broadcasting.
//    @inlinable @inline(__always)
//    static func .==(lhs: TensorView, rhs: TensorView) -> TensorView<Bool> {
//        fatalError("Not implemented")
//        // FunctionId.Equal
////        return Raw.equal(lhs, rhs)
//    }
//
//    /// Computes `lhs != rhs` element-wise and returns a `TensorView` of Boolean
//    /// scalars.
//    /// - Note: `.!=` supports broadcasting.
//    @inlinable @inline(__always)
//    static func .!=(lhs: TensorView, rhs: TensorView) -> TensorView<Bool> {
//        fatalError("Not implemented")
//        // FunctionId.NotEqual
////        return Raw.notEqual(lhs, rhs)
//    }
//
//    /// Computes `lhs == rhs` element-wise and returns a `TensorView` of Boolean
//    /// scalars.
//    /// - Note: `.==` supports broadcasting.
//    @inlinable @inline(__always)
//    static func .==(lhs: Scalar, rhs: TensorView) -> TensorView<Bool> {
//        return TensorView(lhs) .== rhs
//    }
//
//    /// Computes `lhs != rhs` element-wise and returns a `TensorView` of Boolean
//    /// scalars.
//    /// - Note: `.!=` supports broadcasting.
//    @inlinable @inline(__always)
//    static func .!=(lhs: Scalar, rhs: TensorView) -> TensorView<Bool> {
//        return TensorView(lhs) .!= rhs
//    }
//
//    /// Computes `lhs == rhs` element-wise and returns a `TensorView` of Boolean
//    /// scalars.
//    /// - Note: `.==` supports broadcasting.
//    @inlinable @inline(__always)
//    static func .==(lhs: TensorView, rhs: Scalar) -> TensorView<Bool> {
//        return lhs .== TensorView(rhs)
//    }
//
//    /// Computes `lhs != rhs` element-wise and returns a `TensorView` of Boolean
//    /// scalars.
//    /// - Note: `.!=` supports broadcasting.
//    @inlinable @inline(__always)
//    static func .!=(lhs: TensorView, rhs: Scalar) -> TensorView<Bool> {
//        return lhs .!= TensorView(rhs)
//    }
//}
//
//infix operator ≈ : ComparisonPrecedence
//
//public extension TensorView where Scalar : FloatingPoint & Equatable {
//    /// Returns a `TensorView` of Boolean values indicating whether the elements of
//    /// `self` are approximately equal to those of `other`.
//    @inlinable @inline(__always)
//    func elementsApproximatelyEqual(_ other: TensorView,
//                                    tolerance: Double = 0.00001) -> TensorView<Bool> {
//        fatalError("Not implemented")
//        // FunctionId.ApproximateEqual
////        return Raw.approximateEqual(self, other, tolerance: tolerance)
//    }
//}
//
//public extension TensorView where Scalar == Bool {
//    /// Computes `!self` element-wise.
//    @inlinable @inline(__always)
//    func elementsLogicalNot() -> TensorView {
//        fatalError("Not implemented")
//        // FunctionId.LogicalNot
////        return Raw.logicalNot(self)
//    }
//
//    /// Computes `self && other` element-wise.
//    /// - Note: `&&` supports broadcasting.
//    @inlinable @inline(__always)
//    func elementsLogicalAnd(_ other: TensorView) -> TensorView {
//        fatalError("Not implemented")
//        // FunctionId.LogicalAnd
////        return Raw.logicalAnd(self, other)
//    }
//
//    /// Computes `self && other` element-wise, broadcasting `other`.
//    @inlinable @inline(__always)
//    func elementsLogicalAnd(_ other: Scalar) -> TensorView {
//        return elementsLogicalAnd(TensorView(other))
//    }
//
//    /// Computes `self || other` element-wise.
//    @inlinable @inline(__always)
//    func elementsLogicalOr(_ other: TensorView) -> TensorView {
//        fatalError("Not implemented")
//        // FunctionId.LogicalOr
////        return Raw.logicalOr(self, other)
//    }
//
//    /// Computes `self || other` element-wise, broadcasting `other`.
//    @inlinable @inline(__always)
//    func elementsLogicalOr(_ other: Scalar) -> TensorView {
//        return elementsLogicalOr(TensorView(other))
//    }
//}
//
//// TODO
////==============================================================================
//// Transforms
////public extension TensorView {
////    /// Returns a transposed TensorView, with dimensions permuted in the specified
////    /// order.
////    @inlinable @inline(__always)
////    @differentiable(
////    wrt: self, vjp: _vjpTransposed(withPermutations:)
////    where Scalar : TensorFlowFloatingPoint
////    )
////    func transposed(
////        withPermutations permutations: TensorView<Int32>
////        ) -> TensorView {
////        return Raw.transpose(self, perm: permutations)
////    }
////
////    /// Returns a transposed TensorView, with dimensions permuted in the specified
////    /// order.
////    @inlinable @inline(__always)
////    @differentiable(
////    wrt: self, vjp: _vjpTransposed(withPermutations:)
////    where Scalar : TensorFlowFloatingPoint
////    )
////    func transposed(withPermutations permutations: [Int32]) -> TensorView {
////        return transposed(withPermutations: TensorView<Int32>(permutations))
////    }
////
////    /// Returns a transposed TensorView, with dimensions permuted in the specified
////    /// order.
////    @inlinable @inline(__always)
////    @differentiable(
////    wrt: self, vjp: _vjpTransposed(withPermutations:)
////    where Scalar : TensorFlowFloatingPoint
////    )
////    func transposed(withPermutations permutations: Int32...) -> TensorView {
////        return transposed(withPermutations: permutations)
////    }
////
////    /// Returns a transposed TensorView, with dimensions permuted in reverse order.
////    @inlinable @inline(__always)
////    @differentiable(
////    wrt: self, vjp: _vjpTransposed()
////    where Scalar : TensorFlowFloatingPoint
////    )
////    func transposed() -> TensorView {
////        let defaultPermutations = rankTensor - 1 - TensorView<Int32>(
////            rangeFrom: 0, to: rank, stride: 1
////        )
////        return transposed(withPermutations: TensorView<Int32>(defaultPermutations))
////    }
////}
////
////==============================================================================
//// Concatenate
//public extension TensorView {
//    /// Concatenates tensors along the specified axis.
//    /// - Precondition: The tensors must have the same dimensions, except for the
//    ///   specified axis.
//    /// - Precondition: The axis must be in the range `-rank..<rank`.
//    @inlinable @inline(__always)
////    @differentiable(vjp: _vjpConcatenated where Scalar : TensorFlowFloatingPoint)
//    func concatenated(with other: TensorView, alongAxis axis: Int32 = 0) -> TensorView {
//        fatalError("Not implemented")
//        // FunctionId.Concatentate
////        return Raw.concatV2([self, other], axis: TensorView<Int32>(axis))
//    }
//
//    /// Concatenation operator.
//    /// - Note: `++` is a custom operator that does not exist in Swift, but does
//    ///   in Haskell/Scala. Its addition is not an insignificant language change
//    ///   and may be controversial. The existence/naming of `++` will be discussed
//    ///   during a later API design phase.
//    @inlinable @inline(__always)
////    @differentiable(where Scalar : TensorFlowFloatingPoint)
//    static func ++ (lhs: TensorView, rhs: TensorView) -> TensorView {
//        return lhs.concatenated(with: rhs)
//    }
//}
//
////internal extension TensorView where Scalar : TensorFlowFloatingPoint {
////    @inlinable @inline(__always)
////    func _vjpConcatenated(with other: TensorView, alongAxis axis: Int32)
////        -> (TensorView, (TensorView) -> (TensorView, TensorView)) {
////            let idx = axis >= 0 ? axis : rank - axis
////            let splits = TensorView<Int32>([shapeTensor[idx], other.shapeTensor[idx]])
////            return (Raw.concatV2([self, other], axis: TensorView<Int32>(axis)), { result in
////                let ret: (TensorHandle<Scalar>, TensorHandle<Scalar>) = #tfop("SplitV",
////                                                                              result,
////                                                                              splits,
////                                                                              TensorView<Int32>(axis),
////                                                                              num_split: Int64(2),
////                                                                              T$dtype: Scalar.tensorFlowDataType,
////                                                                              Tlen$dtype: Int32.tensorFlowDataType)
////                return (TensorView(handle: ret.0), TensorView(handle: ret.1))
////            })
////    }
////}
//
//==============================================================================
// Element-wise math functions

// Export Glibc/Darwin math functions. We should not require users to import
// Foundation/Darwin/Glibc in order to use scalar math functions.
//
#if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
@_exported import Darwin.C
#else
@_exported import Glibc
#endif
//
// FIXME(rxwei): Scoped imports are not yet supported in parseable module
// interfaces, so `@_exported import` won't work. When that becomes supported,
// switch to `@_exported import` by removing `import Darwin.C/Glibc` above and
// uncommenting the following lines. In the meantime, consider using indirect
// wrappers for each function so that random libc symbols won't be leaked to
// users' code completion.
//
// #if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
// @_exported import func Darwin.C.sin
// @_exported import func Darwin.C.cos
// @_exported import func Darwin.C.tan
// @_exported import func Darwin.C.sinf
// @_exported import func Darwin.C.cosf
// @_exported import func Darwin.C.tanf
// @_exported import func Darwin.C.sinh
// @_exported import func Darwin.C.cosh
// @_exported import func Darwin.C.tanh
// @_exported import func Darwin.C.sinhf
// @_exported import func Darwin.C.coshf
// @_exported import func Darwin.C.tanhf
// @_exported import func Darwin.C.log
// @_exported import func Darwin.C.logf
// @_exported import func Darwin.C.exp
// @_exported import func Darwin.C.expf
// @_exported import func Darwin.C.pow
// @_exported import func Darwin.C.powf
// #else
// @_exported import func Glibc.sin
// @_exported import func Glibc.cos
// @_exported import func Glibc.tan
// @_exported import func Glibc.sinf
// @_exported import func Glibc.cosf
// @_exported import func Glibc.tanf
// @_exported import func Glibc.sinh
// @_exported import func Glibc.cosh
// @_exported import func Glibc.tanh
// @_exported import func Glibc.sinhf
// @_exported import func Glibc.coshf
// @_exported import func Glibc.tanhf
// @_exported import func Glibc.log
// @_exported import func Glibc.logf
// @_exported import func Glibc.exp
// @_exported import func Glibc.expf
// @_exported import func Glibc.pow
// @_exported import func Glibc.powf
// #endif

// SignedNumeric
public extension TensorDataView where Scalar: SignedNumeric {
    //==========================================================================
    /// Computes the negation of the specified TensorView element-wise.
    @inlinable @inline(__always)
    //    @differentiable(
    //    vjp: _vjpNegate(_:)
    //    where Scalar : TensorFlowFloatingPoint
    //    )
    static func neg(_ x: Self, result: inout Self,
                    using deviceStream: DeviceStream? = nil) throws {
        let stream = deviceStream ?? _ThreadLocal.value.defaultStream
        try stream.neg(x: x, result: &result)
    }
    
    // returns new view
    @inlinable @inline(__always)
    //    @differentiable(
    //    vjp: _vjpNegate(_:)
    //    where Scalar : TensorFlowFloatingPoint
    //    )
    static func neg(_ x: Self, using stream: DeviceStream? = nil) throws -> Self {
        var result = Self.init(denseLike: x)
        try neg(x, result: &result, using: stream)
        return result
    }
    
    // operator
    @inlinable @inline(__always)
    //    @differentiable(
    //    vjp: _vjpNegate(_:)
    //    where Scalar : TensorFlowFloatingPoint
    //    )
    static prefix func - (rhs: Self) -> Self {
        return _ThreadLocal.value.catchError {
            return try neg(rhs)
        }
    }

    //==========================================================================
    /// Computes the absolute value of the specified TensorView element-wise.
    @inlinable @inline(__always)
    //@differentiable(vjp: _vjpAbs(_:) where T : TensorFlowFloatingPoint)
    static func abs(_ x: Self, result: inout Self,
                    using deviceStream: DeviceStream? = nil) throws {
        let stream = deviceStream ?? _ThreadLocal.value.defaultStream
        try stream.abs(x: x, result: &result)
    }
    
    // returns new view
    @inlinable @inline(__always)
    //@differentiable(vjp: _vjpAbs(_:) where T : TensorFlowFloatingPoint)
    static func abs(_ x: Self, using stream: DeviceStream? = nil) throws -> Self {
        var result = Self.init(denseLike: x)
        try abs(x, result: &result, using: stream)
        return result
    }
}

//==========================================================================
/// Computes the natural logarithm of the specified TensorView element-wise.
@inlinable @inline(__always)
//@differentiable(vjp: _vjpLog(_:) where T : TensorFlowFloatingPoint)
public func log<T>(_ x: T, result: inout T,
                using deviceStream: DeviceStream? = nil) throws
    where T: TensorDataView, T.Scalar: FloatingPoint {
    let stream = deviceStream ?? _ThreadLocal.value.defaultStream
    try stream.log(x: x, result: &result)
}

// returns new view
@inlinable @inline(__always)
//@differentiable(vjp: _vjpLog(_:) where T : TensorFlowFloatingPoint)
public func log<T>(_ x: T, using stream: DeviceStream? = nil) throws -> T
    where T: TensorDataView, T.Scalar: FloatingPoint {
    var result = T.init(denseLike: x)
    try log(x, result: &result, using: stream)
    return result
}

//==========================================================================
/// Computes `sin` of the specified TensorView element-wise.
@inlinable @inline(__always)
//@differentiable(vjp: _vjpSin(_:) where T : TensorFlowFloatingPoint)
public func sin<T>(_ x: T, result: inout T,
                using deviceStream: DeviceStream? = nil) throws
    where T: TensorDataView, T.Scalar: FloatingPoint {
    let stream = deviceStream ?? _ThreadLocal.value.defaultStream
    try stream.sin(x: x, result: &result)
}

// returns new view
@inlinable @inline(__always)
//@differentiable(vjp: _vjpSin(_:) where T : TensorFlowFloatingPoint)
public func sin<T>(_ x: T, using stream: DeviceStream? = nil) throws -> T
    where T: TensorDataView, T.Scalar: FloatingPoint {
    var result = T.init(denseLike: x)
    try sin(x, result: &result, using: stream)
    return result
}

//==========================================================================
/// Computes `cos` of the specified TensorView element-wise.
@inlinable @inline(__always)
//@differentiable(vjp: _vjpCos(_:) where T : TensorFlowFloatingPoint)
public func cos<T>(_ x: T, result: inout T,
                using deviceStream: DeviceStream? = nil) throws
    where T: TensorDataView, T.Scalar: FloatingPoint {
    let stream = deviceStream ?? _ThreadLocal.value.defaultStream
    try stream.cos(x: x, result: &result)
}

// returns new view
@inlinable @inline(__always)
//@differentiable(vjp: _vjpCos(_:) where T : TensorFlowFloatingPoint)
public func cos<T>(_ x: T, using stream: DeviceStream? = nil) throws -> T
    where T: TensorDataView, T.Scalar: FloatingPoint {
    var result = T.init(denseLike: x)
    try cos(x, result: &result, using: stream)
    return result
}

//==========================================================================
/// Computes `tan` of the specified TensorView element-wise.
@inlinable @inline(__always)
//@differentiable(vjp: _vjpTan(_:) where T : TensorFlowFloatingPoint)
public func tan<T>(_ x: T, result: inout T,
                using deviceStream: DeviceStream? = nil) throws
    where T: TensorDataView, T.Scalar: FloatingPoint {
    let stream = deviceStream ?? _ThreadLocal.value.defaultStream
    try stream.tan(x: x, result: &result)
}

// returns new view
@inlinable @inline(__always)
//@differentiable(vjp: _vjpTan(_:) where T : TensorFlowFloatingPoint)
public func tan<T>(_ x: T, using stream: DeviceStream? = nil) throws -> T
    where T: TensorDataView, T.Scalar: FloatingPoint {
    var result = T.init(denseLike: x)
    try tan(x, result: &result, using: stream)
    return result
}

//==========================================================================
/// Computes `sinh` of the specified TensorView element-wise.
@inlinable @inline(__always)
//@differentiable(vjp: _vjpSinh(_:) where T : TensorFlowFloatingPoint)
public func sinh<T>(_ x: T, result: inout T,
                 using deviceStream: DeviceStream? = nil) throws
    where T: TensorDataView, T.Scalar: FloatingPoint {
    let stream = deviceStream ?? _ThreadLocal.value.defaultStream
    try stream.sinh(x: x, result: &result)
}

// returns new view
@inlinable @inline(__always)
//@differentiable(vjp: _vjpSinh(_:) where T : TensorFlowFloatingPoint)
public func sinh<T>(_ x: T, using stream: DeviceStream? = nil) throws -> T
    where T: TensorDataView, T.Scalar: FloatingPoint {
    var result = T.init(denseLike: x)
    try sinh(x, result: &result, using: stream)
    return result
}

//==========================================================================
/// Computes `cosh` of the specified TensorView element-wise.
@inlinable @inline(__always)
//@differentiable(vjp: _vjpCosh(_:) where T : TensorFlowFloatingPoint)
public func cosh<T>(_ x: T, result: inout T,
                 using deviceStream: DeviceStream? = nil) throws
    where T: TensorDataView, T.Scalar: FloatingPoint {
    let stream = deviceStream ?? _ThreadLocal.value.defaultStream
    try stream.cosh(x: x, result: &result)
}

// returns new view
@inlinable @inline(__always)
//@differentiable(vjp: _vjpCosh(_:) where T : TensorFlowFloatingPoint)
public func cosh<T>(_ x: T, using stream: DeviceStream? = nil) throws -> T
    where T: TensorDataView, T.Scalar: FloatingPoint {
    var result = T.init(denseLike: x)
    try cosh(x, result: &result, using: stream)
    return result
}

//==========================================================================
/// Computes `tanh` of the specified TensorView element-wise.
@inlinable @inline(__always)
//@differentiable(vjp: _vjpTanh(_:) where T : TensorFlowFloatingPoint)
public func tanh<T>(_ x: T, result: inout T,
                 using deviceStream: DeviceStream? = nil) throws
    where T: TensorDataView, T.Scalar: FloatingPoint {
    let stream = deviceStream ?? _ThreadLocal.value.defaultStream
    try stream.tanh(x: x, result: &result)
}

// returns new view
@inlinable @inline(__always)
//@differentiable(vjp: _vjpTanh(_:) where T : TensorFlowFloatingPoint)
public func tanh<T>(_ x: T, using stream: DeviceStream? = nil) throws -> T
    where T: TensorDataView, T.Scalar: FloatingPoint {
    var result = T.init(denseLike: x)
    try tanh(x, result: &result, using: stream)
    return result
}

//==========================================================================
/// Computes the square root of the specified TensorView element-wise.
@inlinable @inline(__always)
//@differentiable(vjp: _vjpSqrt(_:) where T : TensorFlowFloatingPoint)
public func sqrt<T>(_ x: T, result: inout T,
                 using deviceStream: DeviceStream? = nil) throws
    where T: TensorDataView, T.Scalar: FloatingPoint {
    let stream = deviceStream ?? _ThreadLocal.value.defaultStream
    try stream.sqrt(x: x, result: &result)
}

// returns new view
@inlinable @inline(__always)
//@differentiable(vjp: _vjpSqrt(_:) where T : TensorFlowFloatingPoint)
public func sqrt<T>(_ x: T, using stream: DeviceStream? = nil) throws -> T
    where T: TensorDataView, T.Scalar: FloatingPoint {
    var result = T.init(denseLike: x)
    try sqrt(x, result: &result, using: stream)
    return result
}

//==========================================================================
/// Computes the inverse square root of the specified TensorView element-wise.
@inlinable @inline(__always)
//@differentiable(vjp: _vjpRsqrt(_:) where T : TensorFlowFloatingPoint)
public func rsqrt<T>(_ x: T, result: inout T,
                     using deviceStream: DeviceStream? = nil) throws
    where T: TensorDataView, T.Scalar: FloatingPoint {
        let stream = deviceStream ?? _ThreadLocal.value.defaultStream
        try stream.rsqrt(x: x, result: &result)
}

// returns new view
@inlinable @inline(__always)
//@differentiable(vjp: _vjpRsqrt(_:) where T : TensorFlowFloatingPoint)
public func rsqrt<T>(_ x: T, using stream: DeviceStream? = nil) throws -> T
    where T: TensorDataView, T.Scalar: FloatingPoint {
        var result = T.init(denseLike: x)
        try rsqrt(x, result: &result, using: stream)
        return result
}

//==========================================================================
/// Computes `exp` of the specified TensorView element-wise.
@inlinable @inline(__always)
//@differentiable(vjp: _vjpExp(_:) where T : TensorFlowFloatingPoint)
public func exp<T>(_ x: T, result: inout T,
                   using deviceStream: DeviceStream? = nil) throws
    where T: TensorDataView, T.Scalar: FloatingPoint {
        let stream = deviceStream ?? _ThreadLocal.value.defaultStream
        try stream.exp(x: x, result: &result)
}

// returns new view
@inlinable @inline(__always)
//@differentiable(vjp: _vjpExp(_:) where T : TensorFlowFloatingPoint)
public func exp<T>(_ x: T, using stream: DeviceStream? = nil) throws -> T
    where T: TensorDataView, T.Scalar: FloatingPoint {
        var result = T.init(denseLike: x)
        try exp(x, result: &result, using: stream)
        return result
}

//==========================================================================
/// Computes the ceiling of the specified TensorView element-wise.
@inlinable @inline(__always)
//@differentiable(vjp: _vjpCeil(_:) where T : TensorFlowFloatingPoint)
public func ceil<T>(_ x: T, result: inout T,
                    using deviceStream: DeviceStream? = nil) throws
    where T: TensorDataView, T.Scalar: FloatingPoint {
        let stream = deviceStream ?? _ThreadLocal.value.defaultStream
        try stream.ceil(x: x, result: &result)
}

// returns new view
@inlinable @inline(__always)
//@differentiable(vjp: _vjpCeil(_:) where T : TensorFlowFloatingPoint)
public func ceil<T>(_ x: T, using stream: DeviceStream? = nil) throws -> T
    where T: TensorDataView, T.Scalar: FloatingPoint {
        var result = T.init(denseLike: x)
        try ceil(x, result: &result, using: stream)
        return result
}

//==========================================================================
/// Computes the floor of the specified TensorView element-wise.
@inlinable @inline(__always)
//@differentiable(vjp: _vjpFloor(_:) where T : TensorFlowFloatingPoint)
public func floor<T>(_ x: T, result: inout T,
                     using deviceStream: DeviceStream? = nil) throws
    where T: TensorDataView, T.Scalar: FloatingPoint {
        let stream = deviceStream ?? _ThreadLocal.value.defaultStream
        try stream.floor(x: x, result: &result)
}

// returns new view
@inlinable @inline(__always)
//@differentiable(vjp: _vjpFloor(_:) where T : TensorFlowFloatingPoint)
public func floor<T>(_ x: T, using stream: DeviceStream? = nil) throws -> T
    where T: TensorDataView, T.Scalar: FloatingPoint {
        var result = T.init(denseLike: x)
        try floor(x, result: &result, using: stream)
        return result
}

//==========================================================================
/// Computes the power of the scalar to the TensorView, broadcasting the scalar.
@inlinable @inline(__always)
// @differentiable(where T : TensorFlowFloatingPoint)
public func pow<T>(_ lhs: T.Scalar, _ rhs: T, result: inout T,
                   using deviceStream: DeviceStream? = nil) throws
    where T: TensorDataView, T.Scalar: FloatingPoint {
        let stream = deviceStream ?? _ThreadLocal.value.defaultStream
        try stream.pow(x: ScalarTensor(value: lhs), y: rhs, result: &result)
}

// returns new view
@inlinable @inline(__always)
// @differentiable(where T : TensorFlowFloatingPoint)
public func pow<T>(_ lhs: T.Scalar, _ rhs: T,
                   using stream: DeviceStream? = nil) throws -> T
    where T: TensorDataView, T.Scalar: FloatingPoint {
        var result = T.init(denseLike: rhs)
        try pow(lhs, rhs, result: &result, using: stream)
        return result
}

//==========================================================================
/// Computes the power of the TensorView to the scalar, broadcasting the scalar.
@inlinable @inline(__always)
// @differentiable(where T : TensorFlowFloatingPoint)
public func pow<T>(_ lhs: T, _ rhs: T.Scalar, result: inout T,
                   using deviceStream: DeviceStream? = nil) throws
    where T: TensorDataView, T.Scalar: FloatingPoint {
        let stream = deviceStream ?? _ThreadLocal.value.defaultStream
        try stream.pow(x: lhs, y: ScalarTensor(value: rhs), result: &result)
}

// returns new view
@inlinable @inline(__always)
// @differentiable(where T : TensorFlowFloatingPoint)
public func pow<T>(_ lhs: T, _ rhs: T.Scalar,
                   using stream: DeviceStream? = nil) throws -> T
    where T: TensorDataView, T.Scalar: FloatingPoint {
        var result = T.init(denseLike: lhs)
        try pow(lhs, rhs, result: &result, using: stream)
        return result
}

//==========================================================================
/// Computes the power of the first TensorView to the second TensorView.
@inlinable @inline(__always)
//@differentiable(vjp: _vjpPow(_:_:) where T : TensorFlowFloatingPoint)
public func pow<T>(_ x: T, _ y: T, result: inout T,
                   using deviceStream: DeviceStream? = nil) throws
    where T: TensorDataView, T.Scalar: FloatingPoint {
        let stream = deviceStream ?? _ThreadLocal.value.defaultStream
        try stream.pow(x: x, y: y, result: &result)
}

// returns new view
@inlinable @inline(__always)
//@differentiable(vjp: _vjpPow(_:_:) where T : TensorFlowFloatingPoint)
public func pow<T>(_ x: T, _ y: T, using stream: DeviceStream? = nil) throws -> T
    where T: TensorDataView, T.Scalar: FloatingPoint {
        var result = T.init(denseLike: x)
        try pow(x, y, result: &result, using: stream)
        return result
}


///// Computes the element-wise maximum of two tensors.
///// - Note: `max` supports broadcasting.
//@inlinable @inline(__always)
////@differentiable(vjp: _vjpMax(_:_:) where T : TensorFlowFloatingPoint)
//public func max<T>(_ lhs: TensorView<T>, _ rhs: TensorView<T>) -> TensorView<T>
//    where T : Numeric & Comparable {
//        fatalError("Not implemented")
//        // FunctionId.
////        return Raw.maximum(lhs, rhs)
//}
//
///// Computes the element-wise maximum of the scalar and the TensorView, broadcasting
///// the scalar.
//@inlinable @inline(__always)
////@differentiable(where T : TensorFlowFloatingPoint)
//public func max<T>(_ lhs: T, _ rhs: TensorView<T>) -> TensorView<T>
//    where T : Numeric & Comparable {
//        return max(TensorView(lhs), rhs)
//}
//
///// Computes the element-wise maximum of the scalar and the TensorView, broadcasting
///// the scalar.
//@inlinable @inline(__always)
//// @differentiable(where T : TensorFlowFloatingPoint)
//public func max<T>(_ lhs: TensorView<T>, _ rhs: T) -> TensorView<T>
//    where T : Numeric & Comparable {
//        return max(lhs, TensorView(rhs))
//}
//
///// Computes the element-wise minimum of two tensors.
///// - Note: `min` supports broadcasting.
//@inlinable @inline(__always)
////@differentiable(vjp: _vjpMin(_:_:) where T : TensorFlowFloatingPoint)
//public func min<T>(_ lhs: TensorView<T>, _ rhs: TensorView<T>) -> TensorView<T>
//    where T : Numeric & Comparable {
//        fatalError("Not implemented")
//        // FunctionId.
////        return Raw.minimum(lhs, rhs)
//}
//
///// Computes the element-wise minimum of the scalar and the TensorView, broadcasting
///// the scalar.
//@inlinable @inline(__always)
//// @differentiable(where T : TensorFlowFloatingPoint)
//public func min<T>(_ lhs: T, _ rhs: TensorView<T>) -> TensorView<T>
//    where T : Numeric & Comparable {
//        return min(TensorView(lhs), rhs)
//}
//
///// Computes the element-wise minimum of the scalar and the TensorView, broadcasting
///// the scalar.
//@inlinable @inline(__always)
//// @differentiable(where T : TensorFlowFloatingPoint)
//public func min<T>(_ lhs: TensorView<T>, _ rhs: T) -> TensorView<T>
//    where T : Numeric & Comparable {
//        return min(lhs, TensorView(rhs))
//}
//
///// Computes the square of the TensorView.
//public extension TensorView where Scalar : Numeric {
//    @inlinable @inline(__always)
////    @differentiable(
////    wrt: self, vjp: _vjpSquared()
////    where Scalar : TensorFlowFloatingPoint
////    )
//    func squared() -> TensorView {
//        fatalError("Not implemented")
//        // FunctionId.
////        return Raw.square(self)
//    }
//}
//
///// Computes the log-softmax of the specified TensorView element-wise.
//@inlinable @inline(__always)
////@differentiable(vjp: _vjpLogSoftmax(_:) where T : TensorFlowFloatingPoint)
//public func logSoftmax<T : FloatingPoint>(_ x: TensorView<T>) -> TensorView<T> {
//    fatalError("Not implemented")
//    // FunctionId.
////    return Raw.logSoftmax(logits: x)
//}

////==============================================================================
//// Selection
//
//public extension TensorView where Scalar == Bool {
//    /// Returns a new TensorView containing elements from either `left` or `right`,
//    /// depending on the elements of `self`.
//    ///
//    /// `self` acts as a mask that chooses, based on the value at each scalar,
//    ///  whether the corresponding scalar in the output should be taken from
//    /// `left` (if `true`) or `right` (if `false`).
//    ///
//    /// - Precondition: `left` and `right` must have the same shape. If
//    ///   `left` and `right` are scalar, then `self` must also be scalar. If
//    ///   `left` and `right` have rank greater than or equal to 1, then `self`
//    ///   must be either have the same shape as `left` or be a 1-D `TensorView` such
//    ///   that `self.scalarCount == left[0]`.
//    @available(*, deprecated, message: "Use '.replacing(with:mask:)' instead")
//    @inlinable
//    func selecting<T>(_ left: TensorView<T>, _ right: TensorView<T>) -> TensorView<T> {
//        return left.replacing(with: right, where: self)
//    }
//}
//
//public extension TensorView {
//    /// Replaces elements of this TensorView with `other` in the lanes where `mask` is
//    /// `true`.
//    ///
//    /// - Precondition: `self` and `other` must have the same shape. If
//    ///   `self` and `other` are scalar, then `mask` must also be scalar. If
//    ///   `self` and `other` have rank greater than or equal to `1`, then `mask`
//    ///   must be either have the same shape as `self` or be a 1-D `TensorView` such
//    ///   that `mask.scalarCount == self.shape[0]`.
//    @inlinable
////    @differentiable(wrt: (self, other),
////    vjp: _vjpReplacing where Scalar : TensorFlowFloatingPoint)
//    func replacing(with other: TensorView,
//                   where mask: TensorView<Bool>) -> TensorView {
//        fatalError("Not implemented")
//        // FunctionId.
////        return Raw.select(condition: mask, t: self, e: other)
//    }
//}
//
//public extension TensorView where Scalar : TensorFlowFloatingPoint {
//    @inlinable
//    internal func _vjpReplacing(with other: TensorView, where mask: TensorView<Bool>)
//        -> (TensorView, (TensorView) -> (TensorView, TensorView)) {
//            return (replacing(with: other, where: mask), { v in
//                let zeros = TensorView(zeros: v.shape)
//                return (v.replacing(with: zeros, where: mask),
//                        zeros.replacing(with: v, where: mask))
//            })
//    }
//}
//
////==============================================================================
//// Reduction
//
//public extension TensorView where Scalar == Bool {
//    /// Returns `true` if all scalars are equal to `true`. Otherwise, returns
//    /// `false`.
//    // NOTE: This overload is necessary, otherwise `all()` would refer
//    // to the variadic method `all(squeezingAxes:)` with zero indices.
//    @inlinable @inline(__always)
//    func all() -> Bool {
//        fatalError("Not implemented")
//        // FunctionId.AndAny
////        let axes = TensorView<Int32>(rangeFrom: 0, to: rank, stride: 1)
////        return _TFGetScalarOrDie(Raw.all(self, reductionIndices: axes).handle)
//    }
//
//    /// Returns `true` if any scalars are equal to `true`. Otherwise, returns
//    /// `false`.
//    // NOTE: This overload is necessary, otherwise `any()` would refer
//    // to the variadic method `any(squeezingAxes:)` with zero indices.
//    @inlinable @inline(__always)
//    func any() -> Bool {
//        fatalError("Not implemented")
//        // OrAny
////        let axes = TensorView<Int32>(rangeFrom: 0, to: rank, stride: 1)
////        return _TFGetScalarOrDie(Raw.any(self, reductionIndices: axes).handle)
//    }
//
//    /// Performs a logical AND operation along the specified axes. The reduced
//    /// dimensions are removed.
//    /// - Parameter axes: The dimensions to reduce.
//    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
//    @inlinable @inline(__always)
//    func all(squeezingAxes axes: Int...) -> TensorView {
//        fatalError("Not implemented")
//        // FunctionId.AndAll
////        return Raw.all(self, reductionIndices: TensorView<Int32>(axes), keepDims: false)
//    }
//
//    /// Performs a logical AND operation along the specified axes. The reduced
//    /// dimensions are removed.
//    /// - Parameter axes: The dimensions to reduce.
//    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
//    @inlinable @inline(__always)
//    func any(squeezingAxes axes: Int...) -> TensorView {
//        fatalError("Not implemented")
//        // FunctionId.OrAny
////        return Raw.any(self, reductionIndices: TensorView<Int32>(axes), keepDims: false)
//    }
//
//    /// Performs a logical AND operation along the specified axes. The reduced
//    /// dimensions are retained with value 1.
//    /// - Parameter axes: The dimensions to reduce.
//    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
//    @inlinable @inline(__always)
//    func all(alongAxes axes: Int...) -> TensorView {
//        fatalError("Not implemented")
//        // FunctionId.AndAll
////        return Raw.all(self, reductionIndices: TensorView<Int32>(axes), keepDims: true)
//    }
//
//    /// Performs a logical OR operation along the specified axes. The reduced
//    /// dimensions are retained with value 1.
//    /// - Parameter axes: The dimensions to reduce.
//    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
//    @inlinable @inline(__always)
//    func any(alongAxes axes: Int...) -> TensorView {
//        fatalError("Not implemented")
//        // FunctionId.OrAny
////        return Raw.any(self, reductionIndices: TensorView<Int32>(axes), keepDims: true)
//    }
//}
//
//public extension TensorView where Scalar : Numeric & Comparable {
//    // NOTE: This overload is necessary, otherwise `min()` would refer
//    // to the variadic method `min(squeezingAxes:)` with zero indices.
//    @inlinable @inline(__always)
//    func min() -> TensorView {
//        fatalError("Not implemented")
//        // FunctionId.Min
////        let axes = TensorView<Int32>(rangeFrom: 0, to: rank, stride: 1)
////        return Raw.min(self, reductionIndices: axes)
//    }
//
//    // NOTE: This overload is necessary, otherwise `max()` would refer
//    // to the variadic method `max(squeezingAxes:)` with zero indices.
//    @inlinable @inline(__always)
//    func max() -> TensorView {
//        fatalError("Not implemented")
//        // FunctionId.Max
////        let axes = TensorView<Int32>(rangeFrom: 0, to: rank, stride: 1)
////        return Raw.max(self, reductionIndices: axes)
//    }
//
//    /// Returns the maximum values along the specified axes. The reduced
//    /// dimensions are removed.
//    /// - Parameter axes: The dimensions to reduce.
//    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
//    @inlinable @inline(__always)
//    func max(squeezingAxes axes: [Int32]) -> TensorView {
//        fatalError("Not implemented")
//        // FunctionId.Max
////        return Raw.max(self, reductionIndices: TensorView<Int32>(axes), keepDims: false)
//    }
//
//    /// Returns the maximum values along the specified axes. The reduced
//    /// dimensions are removed.
//    /// - Parameter axes: The dimensions to reduce.
//    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
//    @inlinable @inline(__always)
//    func max(squeezingAxes axes: Int32...) -> TensorView {
//        return max(squeezingAxes: axes)
//    }
//
//    /// Returns the minimum values along the specified axes. The reduced
//    /// dimensions are removed.
//    /// - Parameter axes: The dimensions to reduce.
//    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
//    @inlinable @inline(__always)
//    func min(squeezingAxes axes: [Int32]) -> TensorView {
//        fatalError("Not implemented")
//        // FunctionId.Min
////        return Raw.min(self, reductionIndices: TensorView<Int32>(axes), keepDims: false)
//    }
//
//    /// Returns the minimum values along the specified axes. The reduced
//    /// dimensions are removed.
//    /// - Parameter axes: The dimensions to reduce.
//    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
//    @inlinable @inline(__always)
//    func min(squeezingAxes axes: Int32...) -> TensorView {
//        return min(squeezingAxes: axes)
//    }
//
//    /// Returns the indices of the maximum values along the specified axes. The
//    /// reduced dimensions are removed.
//    /// - Parameter axes: The dimensions to reduce.
//    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
//    @inlinable @inline(__always)
//    func argmax(squeezingAxis axis: Int32) -> TensorView<Int32> {
//        fatalError("Not implemented")
//        // FunctionId.ArgMax
////        return Raw.argMax(self, dimension: TensorView<Int32>(axis))
//    }
//
//    /// Returns the indices of the minimum values along the specified axes. The
//    /// reduced dimensions are removed.
//    /// - Parameter axes: The dimensions to reduce.
//    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
//    @inlinable @inline(__always)
//    func argmin(squeezingAxis axis: Int32) -> TensorView<Int32> {
//        fatalError("Not implemented")
//        // FunctionId.ArgMin
////        return Raw.argMin(self, dimension: TensorView<Int32>(axis))
//    }
//
//    /// Returns the minimum along the specified axes. The reduced dimensions are
//    /// retained with value 1.
//    /// - Parameter axes: The dimensions to reduce.
//    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
//    @inlinable @inline(__always)
//    func min(alongAxes axes: [Int32]) -> TensorView {
//        fatalError("Not implemented")
//        // FunctionId.Min
////        return Raw.min(self, reductionIndices: TensorView<Int32>(axes), keepDims: true)
//    }
//
//    /// Returns the minimum along the specified axes. The reduced dimensions are
//    /// retained with value 1.
//    /// - Parameter axes: The dimensions to reduce.
//    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
//    @inlinable @inline(__always)
//    func min(alongAxes axes: Int32...) -> TensorView {
//        return min(alongAxes: axes)
//    }
//
//    /// Returns the minimum along the specified axes. The reduced dimensions are
//    /// retained with value 1.
//    /// - Parameter axes: The dimensions to reduce.
//    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
//    @inlinable @inline(__always)
//    func max(alongAxes axes: [Int32]) -> TensorView {
//        fatalError("Not implemented")
//        // FunctionId.Max
////        return Raw.max(self, reductionIndices: TensorView<Int32>(axes), keepDims: true)
//    }
//
//    /// Returns the minimum along the specified axes. The reduced dimensions are
//    /// retained with value 1.
//    /// - Parameter axes: The dimensions to reduce.
//    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
//    @inlinable @inline(__always)
//    func max(alongAxes axes: Int32...) -> TensorView {
//        return max(alongAxes: axes)
//    }
//
//    /// Returns the index of the maximum value of the flattened scalars.
//    @inlinable @inline(__always)
//    func argmax() -> TensorView<Int32> {
//        return flattened().argmax(squeezingAxis: 0)
//    }
//
//    /// Returns the index of the minimum value of the flattened scalars.
//    @inlinable @inline(__always)
//    func argmin() -> TensorView<Int32> {
//        return flattened().argmin(squeezingAxis: 0)
//    }
//}
//
//public extension TensorView where Scalar : Numeric {
//    // NOTE: This overload is necessary, otherwise `mean()` would refer
//    // to the variadic method `mean(squeezingAxes:)` with zero indices.
////    @differentiable(
////    wrt: self, vjp: _vjpMean()
////    where Scalar : TensorFlowFloatingPoint
////    )
//    @inlinable @inline(__always)
//    func mean() -> TensorView {
//        fatalError("Not implemented")
//        // FunctionId.Mean
////        let axes = TensorView<Int32>(rangeFrom: 0, to: rank, stride: 1)
////        return Raw.mean(self, reductionIndices: axes)
//    }
//
//    // NOTE: This overload is necessary, otherwise `sum()` would refer
//    // to the variadic method `sum(squeezingAxes:)` with zero indices.
//    @inlinable @inline(__always)
////    @differentiable(
////    wrt: self, vjp: _vjpSum()
////    where Scalar : TensorFlowFloatingPoint
////    )
//    func sum() -> TensorView {
//        fatalError("Not implemented")
//        // FunctionId.Sum
////        let axes = TensorView<Int32>(rangeFrom: 0, to: rank, stride: 1)
////        return Raw.sum(self, reductionIndices: axes)
//    }
//
//    // NOTE: This overload is necessary, otherwise `sum()` would refer
//    // to the variadic method `sum(squeezingAxes:)` with zero indices.
//    @inlinable @inline(__always)
//    func product() -> TensorView {
//        fatalError("Not implemented")
//        // FunctionId.Prod
////        let axes = TensorView<Int32>(rangeFrom: 0, to: rank, stride: 1)
////        return Raw.prod(self, reductionIndices: axes)
//    }
//
//    /// Returns the arithmetic mean along the specified axes. The reduced
//    /// dimensions are removed.
//    /// - Parameter axes: The dimensions to reduce.
//    /// - Precondition: Each value in `axes` must be in the range `-rank...rank`.
//    @inlinable @inline(__always)
//    func mean(squeezingAxes axes: [Int32]) -> TensorView {
//        fatalError("Not implemented")
//        // FunctionId.Mean
////        return Raw.mean(self, reductionIndices: TensorView<Int32>(axes),
////                        keepDims: false)
//    }
//
//    /// Returns the arithmetic mean along the specified axes. The reduced
//    /// dimensions are removed.
//    /// - Parameter axes: The dimensions to reduce.
//    /// - Precondition: Each value in `axes` must be in the range `-rank...rank`.
//    @inlinable @inline(__always)
//    func mean(squeezingAxes axes: Int32...) -> TensorView {
//        return mean(squeezingAxes: axes)
//    }
//
//    /// Returns the sum along the specified axes. The reduced dimensions are
//    /// removed.
//    /// - Parameter axes: The dimensions to reduce.
//    /// - Precondition: Each value in `axes` must be in the range `-rank...rank`.
//    @inlinable @inline(__always)
//    func sum(squeezingAxes axes: [Int32]) -> TensorView {
//        fatalError("Not implemented")
//        // FunctionId.Sum
////        return Raw.sum(self, reductionIndices: TensorView<Int32>(axes), keepDims: false)
//    }
//
//    /// Returns the sum along the specified axes. The reduced dimensions are
//    /// removed.
//    /// - Parameter axes: The dimensions to reduce.
//    /// - Precondition: Each value in `axes` must be in the range `-rank...rank`.
//    @inlinable @inline(__always)
//    func sum(squeezingAxes axes: Int32...) -> TensorView {
//        return sum(squeezingAxes: axes)
//    }
//
//    /// Returns the product along the specified axes. The reduced dimensions are
//    /// removed.
//    /// - Parameter axes: The dimensions to reduce.
//    /// - Precondition: Each value in `axes` must be in the range `-rank...rank`.
//    @inlinable @inline(__always)
//    func product(squeezingAxes axes: [Int32]) -> TensorView {
//        fatalError("Not implemented")
//        // FunctionId.Prod
////        return Raw.prod(self, reductionIndices: TensorView<Int32>(axes),
////                        keepDims: false)
//    }
//
//    /// Returns the product along the specified axes. The reduced dimensions are
//    /// removed.
//    /// - Parameter axes: The dimensions to reduce.
//    /// - Precondition: Each value in `axes` must be in the range `-rank...rank`.
//    @inlinable @inline(__always)
//    func product(squeezingAxes axes: Int32...) -> TensorView {
//        return product(squeezingAxes: axes)
//    }
//
//    /// Returns the arithmetic mean along the specified axes. The reduced
//    /// dimensions are retained with value 1.
//    /// - Parameter axes: The dimensions to reduce.
//    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
//    @inlinable @inline(__always)
////    @differentiable(
////    wrt: self, vjp: _vjpMean(alongAxes:)
////    where Scalar : TensorFlowFloatingPoint
////    )
//    func mean(alongAxes axes: [Int32]) -> TensorView {
//        fatalError("Not implemented")
//        // FunctionId.Mean
////        return Raw.mean(self, reductionIndices: TensorView<Int32>(axes), keepDims: true)
//    }
//
//    /// Returns the arithmetic mean along the specified axes. The reduced
//    /// dimensions are retained with value 1.
//    /// - Parameter axes: The dimensions to reduce.
//    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
//    @inlinable @inline(__always)
////    @differentiable(wrt: self where Scalar : TensorFlowFloatingPoint)
//    func mean(alongAxes axes: Int32...) -> TensorView {
//        return mean(alongAxes: axes)
//    }
//
//    /// Returns the sum along the specified axes. The reduced dimensions are
//    /// retained with value 1.
//    /// - Parameter axes: The dimensions to reduce.
//    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
//    @inlinable @inline(__always)
////    @differentiable(
////    wrt: self, vjp: _vjpSum(alongAxes:)
////    where Scalar : TensorFlowFloatingPoint
////    )
//    func sum(alongAxes axes: [Int32]) -> TensorView {
//        fatalError("Not implemented")
//        // FunctionId.Sum
////        return Raw.sum(self, reductionIndices: TensorView<Int32>(axes), keepDims: true)
//    }
//
//    /// Returns the sum along the specified axes. The reduced dimensions are
//    /// retained with value 1.
//    /// - Parameter axes: The dimensions to reduce.
//    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
//    @inlinable @inline(__always)
////    @differentiable(wrt: self where Scalar : TensorFlowFloatingPoint)
//    func sum(alongAxes axes: Int32...) -> TensorView {
//        return sum(alongAxes: axes)
//    }
//
//    /// Returns the variance along the specified axes. The reduced dimensions are
//    /// retained with value 1. Does not apply Bessel's correction.
//    /// - Parameter axes: The dimensions to reduce.
//    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
//    @inlinable @inline(__always)
////    @differentiable(wrt: self where Scalar : TensorFlowFloatingPoint)
//    func variance(alongAxes axes: Int32...) -> TensorView {
//        return variance(alongAxes: axes)
//    }
//
//    /// Returns the variance along the specified axes. The reduced dimensions are
//    /// retained with value 1. Does not apply Bessel's correction.
//    /// - Parameter axes: The dimensions to reduce.
//    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
//    @inlinable @inline(__always)
////    @differentiable(wrt: self where Scalar : TensorFlowFloatingPoint)
//    func variance(alongAxes axes: [Int32]) -> TensorView {
//        let mean = self.mean(alongAxes: axes)
//        let squaredDiff = (self - mean).squared()
//        return squaredDiff.mean(alongAxes: axes)
//    }
//
//    /// Returns the product along the specified axes. The reduced dimensions are
//    /// retained with value 1.
//    /// - Parameter axes: The dimensions to reduce.
//    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
//    @inlinable @inline(__always)
//    func product(alongAxes axes: [Int32]) -> TensorView {
//        fatalError("Not implemented")
//        // FunctionId.Prod
////        return Raw.prod(self, reductionIndices: TensorView<Int32>(axes), keepDims: true)
//    }
//
//    /// Returns the product along the specified axes. The reduced dimensions are
//    /// retained with value 1.
//    /// - Parameter axes: The dimensions to reduce.
//    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
//    @inlinable @inline(__always)
//    func product(alongAxes axes: Int32...) -> TensorView {
//        return product(alongAxes: axes)
//    }
//}
//
////==============================================================================
//// Broadcasting
//public extension TensorView {
//    @inlinable @inline(__always)
//    func broadcast(toShape shape: DataShape) -> TensorView {
//        fatalError("Not implemented")
//        // FunctionId.Broadcast
////        return Raw.broadcastTo(self, shape: shape)
//    }
//
//    /// Broadcast to the same shape as the specified `TensorView`.
//    /// - Precondition: The specified shape must be compatible for broadcasting.
//    @inlinable @inline(__always)
//    func broadcast<OtherScalar>(like other: TensorView<OtherScalar>) -> TensorView {
//        return broadcast(toShape: other.shape)
//    }
//}

//public extension TensorView where Scalar : Numeric {
//    @inlinable
//    func unbroadcast(toShape otherShape: TensorView<Int32>) -> TensorView {
//        let rankDiff = (rankTensor - otherShape.scalarCountTensor).rankLifted()
//        let ones: TensorView<Int32> = Raw.fill(dims: rankDiff, value: TensorView<Int32>(1))
//        let paddedShape = ones ++ otherShape
//        let nonEqualIndices = paddedShape .!= shapeTensor
//        let broadcastIndices = Raw.where_(nonEqualIndices).flattened()
//        let unbroadcasted: TensorView = Raw.sum(
//            self, reductionIndices: TensorView<Int32>(broadcastIndices), keepDims: false)
//        return Raw.reshape(unbroadcasted, shape: otherShape)
//    }
//
//    @inlinable @inline(__always)
//    func unbroadcast<OtherScalar>(like other: TensorView<OtherScalar>) -> TensorView {
//        return unbroadcast(toShape: other.shapeTensor)
//    }
//
//    @inlinable @inline(__always)
//    func unbroadcast(to shape: TensorShape) -> TensorView {
//        return unbroadcast(toShape: TensorView<Int32>(shape.dimensions))
//    }
//
//    @inlinable @inline(__always)
//    static func .= (lhs: inout TensorView, rhs: TensorView) {
//        lhs = rhs.broadcast(like: lhs)
//    }
//}
