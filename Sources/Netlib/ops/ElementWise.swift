//******************************************************************************
// Created by Edward Connell on 4/3/19
// Copyright © 2019 Connell Research. All rights reserved.
//
import Foundation

//==============================================================================
/// copy
/// copies the elements from `view` to `result`

/// with placement
/// - Parameter view: tensor to be copied
/// - Parameter result: the tensor where the result will be written
@inlinable @inline(__always)
public func copy<T, R>(view: T, result: inout R) where
    T: TensorView, R: TensorView, T.Viewed == R.Viewed
{
    _Streams.current.copy(view: view, result: &result)
}

/// returns new view
/// - Parameter view: tensor to be copied
/// - Returns: a new tensor containing the result
@inlinable @inline(__always)
public func copy<T, R>(view: T) -> R where
    T: TensorView, R: TensorView, T.Viewed == R.Viewed
{
    var result = R(with: view.extents)
    copy(view: view, result: &result)
    return result
}

public extension TensorView {
    /// returns new view
    /// - Returns: a new tensor containing the result
    @inlinable @inline(__always)
    func copy() -> Self {
        var result = Self(with: extents)
        Netlib.copy(view: self, result: &result)
        return result
    }
}

//==============================================================================
/// abs(x)
/// computes the absolute value of `x`

/// with placement
/// - Parameter x: value tensor
/// - Parameter result: the tensor where the result will be written
@inlinable @inline(__always)
//@differentiable(vjp: _vjpLog(_:) where T: TensorFlowFloatingPoint)
public func abs<T>(_ x: T, result: inout T)
    where T: TensorView, T.Viewed: FloatingPoint
{
    _Streams.current.abs(x: x, result: &result)
}

/// returns new view
/// - Parameter x: value tensor
/// - Returns: a new tensor containing the result
@inlinable @inline(__always)
//@differentiable(vjp: _vjpLog(_:) where T: TensorFlowFloatingPoint)
public func abs<T>(_ x: T) -> T
    where T: TensorView, T.Viewed: FloatingPoint
{
    var result = T(with: x.extents)
    abs(x, result: &result)
    return result
}

public extension TensorView where Viewed: FloatingPoint {
    /// returns new view
    /// - Returns: a new tensor containing the result
    @inlinable @inline(__always)
    //@differentiable(vjp: _vjpAbs(_:) where T: TensorFlowFloatingPoint)
    func abs() -> Self {
        var result = Self(with: extents)
        Netlib.abs(self, result: &result)
        return result
    }
}

//==============================================================================
/// log(x)
/// computes the log of `x`

/// with placement
/// - Parameter x: value tensor
/// - Parameter result: the tensor where the result will be written
@inlinable @inline(__always)
//@differentiable(vjp: _vjpLog(_:) where T: TensorFlowFloatingPoint)
public func log<T>(_ x: T, result: inout T)
    where T: TensorView, T.Viewed: AnyFloatingPoint
{
    _Streams.current.log(x: x, result: &result)
}

/// returns new view
/// - Parameter x: value tensor
/// - Returns: a new tensor containing the result
@inlinable @inline(__always)
//@differentiable(vjp: _vjpLog(_:) where T: TensorFlowFloatingPoint)
public func log<T>(_ x: T) -> T
    where T: TensorView, T.Viewed: AnyFloatingPoint
{
    var result = T(with: x.extents)
    log(x, result: &result)
    return result
}

public extension TensorView where Viewed: AnyFloatingPoint {
    /// returns new view
    /// - Returns: a new tensor containing the result
    @inlinable @inline(__always)
    //@differentiable(vjp: _vjpLog(_:) where T: TensorFlowFloatingPoint)
    func log() -> Self {
        var result = Self(with: extents)
        Netlib.log(self, result: &result)
        return result
    }
}

//==============================================================================
/// logSoftmax(x)
/// computes the logSoftmax of `x`

/// with placement
/// - Parameter x: value tensor
/// - Parameter result: the tensor where the result will be written
@inlinable @inline(__always)
//@differentiable(vjp: _vjpLogSoftmax(_:) where T: TensorFlowFloatingPoint)
public func logSoftmax<T>(_ x: T, result: inout T)
    where T: TensorView, T.Viewed: AnyFloatingPoint
{
    _Streams.current.logSoftmax(x: x, result: &result)
}

/// returns new view
/// - Parameter x: value tensor
/// - Returns: a new tensor containing the result
@inlinable @inline(__always)
//@differentiable(vjp: _vjpLogSoftmax(_:) where T: TensorFlowFloatingPoint)
public func logSoftmax<T>(_ x: T) -> T
    where T: TensorView, T.Viewed: AnyFloatingPoint
{
    var result = T.init(with: x.extents)
    logSoftmax(x, result: &result)
    return result
}

public extension TensorView where Element: AnyFloatingPoint {
    /// returns new view
    /// - Returns: a new tensor containing the result
    @inlinable @inline(__always)
    //@differentiable(vjp: _vjpLogSoftmax(_:) where T: TensorFlowFloatingPoint)
    func logSoftmax() throws -> Self {
        var result = Self(with: extents)
        Netlib.logSoftmax(self, result: &result)
        return result
    }
}

//==============================================================================
/// pow(x, y)
/// raises tensor 'x' to the tensor 'y' power

/// with placement
/// - Parameter x: value tensor
/// - Parameter y: exponent tensor. If the extents are smaller than `x` then
///   broadcasting will be performed via repeated indexing.
/// - Parameter result: the tensor where the result will be written
@inlinable @inline(__always)
//@differentiable(vjp: _vjpPow(_:_:) where T: TensorFlowFloatingPoint)
public func pow<T>(_ x: T, _ y: T, result: inout T)
    where T: TensorView, T.Viewed: AnyNumeric
{
    _Streams.current.pow(x: x, y: y, result: &result)
}

/// returns new view
/// - Parameter x: value tensor
/// - Parameter y: exponent tensor. If the extents are smaller than `x` then
///   broadcasting will be performed via repeated indexing.
/// - Returns: a new tensor containing the result
@inlinable @inline(__always)
//@differentiable(vjp: _vjpPow(_:_:) where T: TensorFlowFloatingPoint)
public func pow<T>(_ x: T, _ y: T) -> T
    where T: TensorView, T.Viewed: AnyNumeric
{
    var result = T.init(with: x.extents)
    pow(x, y, result: &result)
    return result
}

public extension TensorView where Element: AnyNumeric {
    /// returns new view
    /// - Parameter y: exponent tensor. If the extents are smaller than `x` then
    ///   broadcasting will be performed via repeated indexing.
    /// - Returns: a new tensor containing the result
    @inlinable @inline(__always)
    //@differentiable(vjp: _vjpPow(_:_:) where T: TensorFlowFloatingPoint)
    func pow(_ y: Self) -> Self{
        var result = Self(with: extents)
        Netlib.pow(self, y, result: &result)
        return result
    }
}
public extension TensorView where Element: AnyNumeric {
    /// returns new view
    /// - Parameter y: exponent tensor. If the extents are smaller than `x` then
    ///   broadcasting will be performed via repeated indexing.
    /// - Returns: a new tensor containing the result
    @inlinable @inline(__always)
    //@differentiable(vjp: _vjpPow(_:_:) where T: TensorFlowFloatingPoint)
    func pow<S: AnyNumeric>(_ y: S) -> Self {
        var result = Self(with: extents)
        Netlib.pow(self, Self(with: Element(any: y)), result: &result)
        return result
    }
}

public extension TensorView where Element: AnyNumeric {
    /// returns new view
    /// - Parameter y: exponent tensor. If the extents are smaller than `x` then
    ///   broadcasting will be performed via repeated indexing.
    /// - Returns: a new tensor containing the result
    @inlinable @inline(__always)
    //@differentiable(vjp: _vjpPow(_:_:) where T: TensorFlowFloatingPoint)
    func pow<S: AnyInteger>(
        _ y: S, using deviceStream: DeviceStream? = nil) -> Self {
        var result = Self(with: extents)
        Netlib.pow(self, Self(with: Element(any: y)), result: &result)
        return result
    }
}

//==============================================================================
/// fill<T>(result:value:
/// fills the view with the specified value
public func fill<T>(_ result: inout T, with value: T.Viewed) where
    T: TensorView
{
    _Streams.current.fill(&result, with: value)
}

public extension TensorView {
    func filled(with value: Element) -> Self {
        var result = Self(with: extents)
        _Streams.current.fill(&result, with: value)
        return result
    }
}

//==============================================================================
/// fillWithIndex(x:startAt:
/// fills the view with the spatial sequential index
public func fillWithIndex<T>(_ result: inout T, startAt index: Int = 0) where
    T: TensorView, T.Viewed: AnyNumeric
{
    _Streams.current.fillWithIndex(&result, startAt: index)
}

public extension TensorView where Element: AnyNumeric {
    func filledWithIndex(startAt index: Int = 0) -> Self {
        var result = Self(with: extents)
        _Streams.current.fillWithIndex(&result, startAt: index)
        return result
    }
}

//==============================================================================
/// Computes `lhs == rhs` element-wise and returns a `TensorView` of Boolean
/// scalars.
public func equal<T>(lhs: T, rhs: T, result: inout T.BoolView) where
    T: TensorView, T.Viewed: Equatable,
    T.BoolView.Element == Bool
{
    _Streams.current.equal(lhs: lhs, rhs: rhs, result: &result)
}

public extension TensorView where
    Element: Equatable,
    BoolView.Element == Bool
{
    /// operator (Self - scalar)
    /// - Parameter lhs: left hand tensor
    /// - Parameter rhs: right hand scalar. If the extents are smaller than
    ///   `lhs` then broadcasting is performed via repeated indexing.
    /// - Returns: a new tensor containing the result
    @inlinable @inline(__always)
    static func .== (lhs: Self, rhs: Self) -> BoolView {
        var result = BoolView(with: lhs.extents)
        equal(lhs: lhs, rhs: rhs, result: &result)
        return result
    }
}
