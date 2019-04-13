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
public func all<T>(_ x: T, alongAxes axes: Vector<TensorIndex>? = nil,
                   result: inout T)
    where T: TensorView, T.Scalar == Bool {
        
        _ThreadLocal.value.catchError { stream in
            try stream.all(x: x, axes: axes, result: &result)
        }
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
            TensorIndex($0)
        }
        // turn into a vector
        let axesVec = Vector<TensorIndex>(scalars: axes)
        var result = Self.init(shapedLike: self)
        Netlib.all(self, alongAxes: axesVec, result: &result)
        return result
    }
    
    @inlinable @inline(__always)
    func all() -> Self {
        var result = Self.init(shapedLike: self)
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
        let axesVec = Vector<TensorIndex>(
            scalars: axes.map {TensorIndex($0)})
        
        var result = Self.init(shapedLike: self)
        Netlib.all(self, alongAxes: axesVec, result: &result)
        return result.squeezed(axes: axes)
    }
}

