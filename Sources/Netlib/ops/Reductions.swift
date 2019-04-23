//******************************************************************************
// Created by Edward Connell on 4/3/19
// Copyright © 2019 Connell Research. All rights reserved.
//
import Foundation

//==============================================================================
/// all(x:alongAxes:)
/// Returns `true` if all scalars are equal to `true` along the specified
/// axes. Otherwise returns `false`. The result extent along the specified
/// axes will be 1. Rank is not reduced.

/// in place
/// - Parameter x: value tensor
/// - Parameter alongAxes: the axes to operate on
/// - Parameter result: the scalar tensor where the result will be written
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
@inlinable @inline(__always)
public func all<T>(_ x: T, alongAxes axes: Vector<IndexScalar>? = nil,
                   result: inout T)
    where T: TensorView, T.Scalar == Bool
{
    _Streams.current.all(x: x, axes: axes, result: &result)
}

/// returns new view
/// - Parameter alongAxes: the axes to operate on
/// - Returns: a new tensor containing the result
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
public extension TensorView where Self.Scalar == Bool {
    @inlinable @inline(__always)
    func all(alongAxes: Int...) -> Self {
        // make sure to handle negative axes
        let axes = shape.makePositive(indices: alongAxes).map {
            IndexScalar($0)
        }
        // turn into a vector
        let axesVec = Vector<IndexScalar>(scalars: axes)
        var result = Self.init(shapedLike: self)
        Netlib.all(self, alongAxes: axesVec, result: &result)
        return result
    }
    
    @inlinable @inline(__always)
    func all() -> Self {
        let extents = [Int](repeating: 1, count: shape.rank)
        var result = Self.init(shapedLike: self, with: extents)
        Netlib.all(self, result: &result)
        return result
    }
    
    /// returns new view
    /// - Parameter alongAxes: the axes to operate on
    /// - Returns: a new NDTensor containing the result
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable @inline(__always)
    func all(squeezingAxes: Int...) -> NDTensor<Scalar> {
        
        let axes = shape.makePositive(indices: squeezingAxes)
        let axesVec = Vector<IndexScalar>(scalars: axes.map { IndexScalar($0) })
        var result = Self.init(shapedLike: self)
        Netlib.all(self, alongAxes: axesVec, result: &result)
        return result.squeezed(axes: axes)
    }
}

//==============================================================================
/// sum(x:alongAxes:)
/// Returns the sum of all scalars are along the specified
/// axes. The result extent along the specified axes will be 1.
/// Rank is not reduced.

/// in place
/// - Parameter x: value tensor
/// - Parameter alongAxes: the axes to operate on
/// - Parameter result: the scalar tensor where the result will be written
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
@inlinable @inline(__always)
public func sum<T>(_ x: T, alongAxes axes: Vector<IndexScalar>? = nil,
                   result: inout T)
    where T: TensorView, T.Scalar: Numeric
{
    _Streams.current.sum(x: x, axes: axes, result: &result)
}

/// returns new view
/// - Parameter alongAxes: the axes to operate on
/// - Returns: a new tensor containing the result
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
public extension TensorView where Self.Scalar: Numeric {
    @inlinable @inline(__always)
    func sum(alongAxes: Int...) -> Self {
        // make sure to handle negative axes
        let axes = shape.makePositive(indices: alongAxes).map {
            IndexScalar($0)
        }
        // turn into a vector
        let axesVec = Vector<IndexScalar>(scalars: axes)
        var result = Self.init(shapedLike: self)
        Netlib.sum(self, alongAxes: axesVec, result: &result)
        return result
    }
    
    @inlinable @inline(__always)
    func sum() -> Self {
        let extents = [Int](repeating: 1, count: shape.rank)
        var result = Self.init(shapedLike: self, with: extents)
        Netlib.sum(self, result: &result)
        return result
    }
    
    /// returns new view
    /// - Parameter alongAxes: the axes to operate on
    /// - Returns: a new NDTensor containing the result
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable @inline(__always)
    func sum(squeezingAxes: Int...) -> NDTensor<Scalar> {
        
        let axes = shape.makePositive(indices: squeezingAxes)
        let axesVec = Vector<IndexScalar>(scalars: axes.map { IndexScalar($0) })
        var result = Self.init(shapedLike: self)
        Netlib.sum(self, alongAxes: axesVec, result: &result)
        return result.squeezed(axes: axes)
    }
}

