//******************************************************************************
// Created by Edward Connell on 4/3/19
// Copyright © 2019 Connell Research. All rights reserved.
//
/// TensorView operators are defined in several forms
/// - in place: the result is written to a tensor provided by the caller
///
/// - return new view: a new result tensor is created and returned. This is
///   less efficient in iterative cases, but convenient for expression
///   composition.
///
/// - operator form: + - * / etc..
///
/// - scalar arg form: one of the arguments might be passed as a scalar and
///   converted to a scalar tensor for the caller as a convenience.
///
/// - scalar type mismatch: a form is provided to allow the user to pass an
///   integer value where a Float or Double is needed.
///   For example:
///     let m = MatrixTensor<Float>()
///     let x = m + 1
import Foundation

//==============================================================================
/// Add tensors

/// add: in place
/// Adds two tensors to produce their sum
/// - Parameter lhs: left hand tensor
/// - Parameter rhs: right hand tensor. If the size is smaller than `lhs` then
/// broadcasting will be performed via modulo indexing.
/// - Parameter result: the tensor where the result will be written
@inlinable @inline(__always)
//  @differentiable(vjp: _vjpAdd(lhs:rhs:) where Scalar : TensorFlowFloatingPoint)
public func add<T>(_ lhs: T, _ rhs: T, result: inout T,
                   using deviceStream: DeviceStream? = nil) throws
    where T: TensorView, T.Scalar: Numeric {
        
    let stream = deviceStream ?? _ThreadLocal.value.defaultStream
    try stream.add(lhs: lhs, rhs: rhs, result: &result)
}

/// add: returning new view
/// Adds two tensors to produce their sum
/// - Parameter lhs: left hand tensor
/// - Parameter rhs: right hand tensor. If the extents are smaller than
///   `lhs` then broadcasting is performed via modulo indexing.
/// - Returns: a new tensor containing the result
@inlinable @inline(__always)
//  @differentiable(vjp: _vjpAdd(lhs:rhs:) where Scalar : TensorFlowFloatingPoint)
public func add<T>(_ lhs: T, _ rhs: T,
                   using deviceStream: DeviceStream? = nil) throws -> T
    where T: TensorView, T.Scalar: Numeric {
        
    var result = T.init(shapedLike: lhs)
    try add(lhs, rhs, result: &result, using: deviceStream)
    return result
}

public extension TensorView where Self.Scalar: Numeric {
    /// add: operator
    /// Adds two tensors to produce their sum
    /// - Parameter lhs: left hand tensor
    /// - Parameter rhs: right hand tensor. If the extents are smaller than
    ///   `lhs` then broadcasting is performed via modulo indexing.
    /// - Returns: a new tensor containing the result
    @inlinable @inline(__always)
    //  @differentiable(vjp: _vjpAdd(lhs:rhs:) where Scalar : TensorFlowFloatingPoint)
    static func + (lhs: Self, rhs: Self) throws -> Self {
        return try add(lhs, rhs)
    }
}

public extension TensorView where Self.Scalar: FloatingPoint & AnyConvertable {
    /// add: operator (Self + scalar)
    /// Adds two tensors to produce their sum
    /// - Parameter lhs: left hand tensor
    /// - Parameter rhs: right hand scalar. If the extents are smaller than
    ///   `lhs` then broadcasting is performed via modulo indexing.
    /// - Returns: a new tensor containing the result
    @inlinable @inline(__always)
    //  @differentiable(vjp: _vjpAdd(lhs:rhs:) where Scalar : TensorFlowFloatingPoint)
    static func +<S: AnyNumeric>(lhs: Self, rhs: S) throws -> Self {
        let scalarTensor = Self.init(asScalar: Scalar(any: rhs))
        return try add(lhs, scalarTensor)
    }
}

//==============================================================================
/// Subtract tensors
