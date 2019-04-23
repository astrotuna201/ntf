//******************************************************************************
// Created by Edward Connell on 4/3/19
// Copyright © 2019 Connell Research. All rights reserved.
//
import Foundation

//==============================================================================
/// log(x)
/// computes the log of `x`

/// in place
/// - Parameter x: value tensor
/// - Parameter result: the tensor where the result will be written
@inlinable @inline(__always)
//@differentiable(vjp: _vjpLog(_:) where T : TensorFlowFloatingPoint)
public func log<T>(_ x: T, result: inout T)
    where T: TensorView, T.Scalar: AnyFloatingPoint
{
    _Streams.current.log(x: x, result: &result)
}

/// returns new view
/// - Parameter x: value tensor
/// - Returns: a new tensor containing the result
@inlinable @inline(__always)
//@differentiable(vjp: _vjpLog(_:) where T : TensorFlowFloatingPoint)
public func log<T>(_ x: T) -> T
    where T: TensorView, T.Scalar: AnyFloatingPoint
{
    var result = T.init(shapedLike: x)
    log(x, result: &result)
    return result
}

public extension TensorView where Self.Scalar: AnyFloatingPoint {
    /// returns new view
    /// - Returns: a new tensor containing the result
    @inlinable @inline(__always)
    //@differentiable(vjp: _vjpLog(_:) where T : TensorFlowFloatingPoint)
    func log() -> Self {
        var result = Self.init(shapedLike: self)
        Netlib.log(self, result: &result)
        return result
    }
}

//==============================================================================
/// logSoftmax(x)
/// computes the logSoftmax of `x`

/// in place
/// - Parameter x: value tensor
/// - Parameter result: the tensor where the result will be written
@inlinable @inline(__always)
//@differentiable(vjp: _vjpLogSoftmax(_:) where T : TensorFlowFloatingPoint)
public func logSoftmax<T>(_ x: T, result: inout T)
    where T: TensorView, T.Scalar: AnyFloatingPoint
{
    _Streams.current.logSoftmax(x: x, result: &result)
}

/// returns new view
/// - Parameter x: value tensor
/// - Returns: a new tensor containing the result
@inlinable @inline(__always)
//@differentiable(vjp: _vjpLogSoftmax(_:) where T : TensorFlowFloatingPoint)
public func logSoftmax<T>(_ x: T) -> T
    where T: TensorView, T.Scalar: AnyFloatingPoint
{
    var result = T.init(shapedLike: x)
    logSoftmax(x, result: &result)
    return result
}

public extension TensorView where Self.Scalar: AnyFloatingPoint {
    /// returns new view
    /// - Returns: a new tensor containing the result
    @inlinable @inline(__always)
    //@differentiable(vjp: _vjpLogSoftmax(_:) where T : TensorFlowFloatingPoint)
    func logSoftmax() throws -> Self {
        var result = Self.init(shapedLike: self)
        Netlib.logSoftmax(self, result: &result)
        return result
    }
}

//==============================================================================
/// pow(x, y)
/// raises tensor 'x' to the tensor 'y' power

/// in place
/// - Parameter x: value tensor
/// - Parameter y: exponent tensor. If the extents are smaller than `x` then
///   broadcasting will be performed via repeated indexing.
/// - Parameter result: the tensor where the result will be written
@inlinable @inline(__always)
//@differentiable(vjp: _vjpPow(_:_:) where T : TensorFlowFloatingPoint)
public func pow<T>(_ x: T, _ y: T, result: inout T)
    where T: TensorView, T.Scalar: AnyNumeric
{
    _Streams.current.pow(x: x, y: y, result: &result)
}

/// returns new view
/// - Parameter x: value tensor
/// - Parameter y: exponent tensor. If the extents are smaller than `x` then
///   broadcasting will be performed via repeated indexing.
/// - Returns: a new tensor containing the result
@inlinable @inline(__always)
//@differentiable(vjp: _vjpPow(_:_:) where T : TensorFlowFloatingPoint)
public func pow<T>(_ x: T, _ y: T) -> T
    where T: TensorView, T.Scalar: AnyNumeric
{
    var result = T.init(shapedLike: x)
    pow(x, y, result: &result)
    return result
}

public extension TensorView where Self.Scalar: AnyNumeric {
    /// returns new view
    /// - Parameter y: exponent tensor. If the extents are smaller than `x` then
    ///   broadcasting will be performed via repeated indexing.
    /// - Returns: a new tensor containing the result
    @inlinable @inline(__always)
    //@differentiable(vjp: _vjpPow(_:_:) where T : TensorFlowFloatingPoint)
    func pow(_ y: Self) -> Self{
        var result = Self.init(shapedLike: self)
        Netlib.pow(self, y, result: &result)
        return result
    }
}
public extension TensorView where Self.Scalar: AnyNumeric {
    /// returns new view
    /// - Parameter y: exponent tensor. If the extents are smaller than `x` then
    ///   broadcasting will be performed via repeated indexing.
    /// - Returns: a new tensor containing the result
    @inlinable @inline(__always)
    //@differentiable(vjp: _vjpPow(_:_:) where T : TensorFlowFloatingPoint)
    func pow<S: AnyNumeric>(_ y: S) -> Self {
        var result = Self.init(shapedLike: self)
        Netlib.pow(self, Self.init(Scalar(any: y)), result: &result)
        return result
    }
}

public extension TensorView where Self.Scalar: AnyNumeric {
    /// returns new view
    /// - Parameter y: exponent tensor. If the extents are smaller than `x` then
    ///   broadcasting will be performed via repeated indexing.
    /// - Returns: a new tensor containing the result
    @inlinable @inline(__always)
    //@differentiable(vjp: _vjpPow(_:_:) where T : TensorFlowFloatingPoint)
    func pow<S: AnyInteger>(
        _ y: S, using deviceStream: DeviceStream? = nil) -> Self {
        var result = Self.init(shapedLike: self)
        Netlib.pow(self, Self.init(Scalar(any: y)), result: &result)
        return result
    }
}

//==============================================================================
/// fillWithIndex(x:startAt:
/// fills the view with the spatial sequential index
public func fillWithIndex<T>(result: inout T, startAt index: Int = 0) where
    T: TensorView, T.Scalar: AnyNumeric
{
    _Streams.current.fillWithIndex(result: &result, startAt: index)
}

public extension TensorView where Scalar: AnyNumeric {
    func filledWithIndex(startAt index: Int = 0) -> Self {
        var result = Self.init(shapedLike: self)
        _Streams.current.fillWithIndex(result: &result, startAt: index)
        return result
    }
}
