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
public func log<T>(_ x: T, result: inout T,
                   using deviceStream: DeviceStream? = nil) throws
    where T: TensorView, T.Scalar: FloatingPoint {
        
        let stream = deviceStream ?? _ThreadLocal.value.defaultStream
        try stream.log(x: x, result: &result)
}

/// returns new view
/// - Parameter x: value tensor
/// - Parameter y: exponent tensor. If the extents are smaller than `x` then
///   broadcasting will be performed via modulo indexing.
/// - Returns: a new tensor containing the result
@inlinable @inline(__always)
//@differentiable(vjp: _vjpLog(_:) where T : TensorFlowFloatingPoint)
public func log<T>(_ x: T, using deviceStream: DeviceStream? = nil) throws -> T
    where T: TensorView, T.Scalar: FloatingPoint {
        
        var result = T.init(shapedLike: x)
        try log(x, result: &result, using: deviceStream)
        return result
}

//==============================================================================
/// logSoftmax(x)
/// computes the logSoftmax of `x`

/// in place
/// - Parameter x: value tensor
/// - Parameter result: the tensor where the result will be written
@inlinable @inline(__always)
//@differentiable(vjp: _vjpLog(_:) where T : TensorFlowFloatingPoint)
public func logSoftmax<T>(_ x: T, result: inout T,
                          using deviceStream: DeviceStream? = nil) throws
    where T: TensorView, T.Scalar: FloatingPoint {
        
        let stream = deviceStream ?? _ThreadLocal.value.defaultStream
        try stream.logSoftmax(x: x, result: &result)
}

/// returns new view
/// - Parameter x: value tensor
/// - Parameter y: exponent tensor. If the extents are smaller than `x` then
///   broadcasting will be performed via modulo indexing.
/// - Returns: a new tensor containing the result
@inlinable @inline(__always)
//@differentiable(vjp: _vjpLog(_:) where T : TensorFlowFloatingPoint)
public func logSoftmax<T>(_ x: T,
                          using deviceStream: DeviceStream? = nil) throws -> T
    where T: TensorView, T.Scalar: FloatingPoint {
        
        var result = T.init(shapedLike: x)
        try logSoftmax(x, result: &result, using: deviceStream)
        return result
}

//==============================================================================
/// pow(x, y)
/// raises tensor 'x' to the tensor 'y' power

/// in place
/// - Parameter x: value tensor
/// - Parameter y: exponent tensor. If the extents are smaller than `x` then
///   broadcasting will be performed via modulo indexing.
/// - Parameter result: the tensor where the result will be written
@inlinable @inline(__always)
//@differentiable(vjp: _vjpPow(_:_:) where T : TensorFlowFloatingPoint)
public func pow<T>(_ x: T, _ y: T, result: inout T,
                   using deviceStream: DeviceStream? = nil) throws
    where T: TensorView, T.Scalar: Numeric {
        
        let stream = deviceStream ?? _ThreadLocal.value.defaultStream
        try stream.pow(x: x, y: y, result: &result)
}

/// returns new view
/// - Parameter x: value tensor
/// - Parameter y: exponent tensor. If the extents are smaller than `x` then
///   broadcasting will be performed via modulo indexing.
/// - Returns: a new tensor containing the result
@inlinable @inline(__always)
//@differentiable(vjp: _vjpPow(_:_:) where T : TensorFlowFloatingPoint)
public func pow<T>(_ x: T, _ y: T,
                   using deviceStream: DeviceStream? = nil) throws -> T
    where T: TensorView, T.Scalar: Numeric {
        
        var result = T.init(shapedLike: x)
        try pow(x, y, result: &result, using: deviceStream)
        return result
}
