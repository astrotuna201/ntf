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
    where T: TensorView, T.Scalar: FloatingPoint {
        
        _ThreadLocalStream.value.catchError { stream in
            try stream.log(x: x, result: &result)
        }
}

/// returns new view
/// - Parameter x: value tensor
/// - Returns: a new tensor containing the result
@inlinable @inline(__always)
//@differentiable(vjp: _vjpLog(_:) where T : TensorFlowFloatingPoint)
public func log<T>(_ x: T) -> T
    where T: TensorView, T.Scalar: FloatingPoint {
        
        var result = T.init(shapedLike: x)
        log(x, result: &result)
        return result
}

public extension TensorView where Self.Scalar: FloatingPoint {
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
    where T: TensorView, T.Scalar: FloatingPoint {
        
        _ThreadLocalStream.value.catchError { stream in
            try stream.logSoftmax(x: x, result: &result)
        }
}

/// returns new view
/// - Parameter x: value tensor
/// - Returns: a new tensor containing the result
@inlinable @inline(__always)
//@differentiable(vjp: _vjpLogSoftmax(_:) where T : TensorFlowFloatingPoint)
public func logSoftmax<T>(_ x: T) -> T
    where T: TensorView, T.Scalar: FloatingPoint {
        
        var result = T.init(shapedLike: x)
        logSoftmax(x, result: &result)
        return result
}

public extension TensorView where Self.Scalar: FloatingPoint {
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
    where T: TensorView, T.Scalar: Numeric {
        _ThreadLocalStream.value.catchError { stream in
            try stream.pow(x: x, y: y, result: &result)
        }
}

/// returns new view
/// - Parameter x: value tensor
/// - Parameter y: exponent tensor. If the extents are smaller than `x` then
///   broadcasting will be performed via repeated indexing.
/// - Returns: a new tensor containing the result
@inlinable @inline(__always)
//@differentiable(vjp: _vjpPow(_:_:) where T : TensorFlowFloatingPoint)
public func pow<T>(_ x: T, _ y: T) -> T
    where T: TensorView, T.Scalar: Numeric {
        
        var result = T.init(shapedLike: x)
        pow(x, y, result: &result)
        return result
}

public extension TensorView where Self.Scalar: Numeric {
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
public extension TensorView where Self.Scalar: FloatingPoint & AnyNumeric {
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

public extension TensorView where Self.Scalar: BinaryInteger & AnyInteger {
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
/// fill(x:with:
/// fills the view with the scalar value
public func fill<T>(x: T, with value: T.Scalar)
    where T: TensorView {
    _ThreadLocalStream.value.catchError { stream in
        try stream.fill(x: x, with: value)
    }
}

public extension TensorView {
    func fill(with value: Scalar) -> Self {
        let result = Self.init(shapedLike: self)
        _ThreadLocalStream.value.catchError { stream in
            try stream.fill(x: result, with: value)
        }
        return result
    }
}

//==============================================================================
/// fillWithIndex(x:startAt:
/// fills the view with the spatial sequential index
public func fillWithIndex<T>(x: T, startAt index: Int = 0) where
    T: TensorView, T.Scalar: AnyNumeric {
        _ThreadLocalStream.value.catchError { stream in
            try stream.fillWithIndex(x: x, startAt: index)
        }
}

public extension TensorView where Scalar: AnyNumeric {
    func filledWithIndex(startAt index: Int = 0) -> Self {
        let result = Self.init(shapedLike: self)
        _ThreadLocalStream.value.catchError { stream in
            try stream.fillWithIndex(x: result, startAt: index)
        }
        return result
    }
}
