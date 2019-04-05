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
public func all<T>(_ x: T,
                   alongAxes axes: VectorTensor<TensorIndex>? = nil,
                   result: inout T,
                   using deviceStream: DeviceStream? = nil) throws
    where T: TensorView, T.Scalar == Bool {
        
        let stream = deviceStream ?? _ThreadLocal.value.defaultStream
        try stream.all(x: x, axes: axes, result: &result)
}

/// returns new view
/// - Parameter alongAxes: the axes to operate on
/// - Returns: a new tensor containing the result
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
public extension TensorView where Self.Scalar == Bool {
    @inlinable @inline(__always)
    func all(alongAxes: Int...,
        using deviceStream: DeviceStream? = nil) throws -> Self {
        // make sure to handle negative axes
        let axes = shape.makePositive(indices: alongAxes).map {
            TensorIndex($0)
        }
        // turn into a vector
        let axesVec = VectorTensor<TensorIndex>(scalars: axes)
        var result = Self.init(shapedLike: self)
        try Netlib.all(self, alongAxes: axesVec,
                       result: &result, using: deviceStream)
        return result
    }
    
    @inlinable @inline(__always)
    func all(using deviceStream: DeviceStream? = nil) throws -> Self {
        
        var result = Self.init(shapedLike: self)
        try Netlib.all(self, result: &result, using: deviceStream)
        return result
    }
    
    /// returns new view
    /// - Parameter alongAxes: the axes to operate on
    /// - Returns: a new NDTensor containing the result
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable @inline(__always)
    func all(squeezingAxes: Int...,
        using deviceStream: DeviceStream? = nil) throws -> NDTensor<Scalar> {
        
        let axes = shape.makePositive(indices: squeezingAxes)
        let axesVec = VectorTensor<TensorIndex>(
            scalars: axes.map {TensorIndex($0)})
        
        var result = Self.init(shapedLike: self)
        try Netlib.all(self, alongAxes: axesVec,
                       result: &result, using: deviceStream)
        
        return result.squeezed(axes: axes)
    }
}

