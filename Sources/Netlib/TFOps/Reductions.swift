//******************************************************************************
// Created by Edward Connell on 4/3/19
// Copyright © 2019 Connell Research. All rights reserved.
//
import Foundation

//==============================================================================
/// all(x)
/// Returns `true` if all scalars are equal to `true`. Otherwise returns `false`

/// in place
/// - Parameter x: value tensor
/// - Parameter result: the scalar tensor where the result will be written
@inlinable @inline(__always)
public func all<T>(_ x: T, result: inout T,
                   using deviceStream: DeviceStream? = nil) throws
    where T: TensorView, T.Scalar == Bool {
        
        let stream = deviceStream ?? _ThreadLocal.value.defaultStream
        try stream.all(x: x, result: &result)
}

/// returns new view
/// - Parameter x: value tensor
/// - Returns: a new tensor containing the result
public extension TensorView where Self.Scalar == Bool {
    @inlinable @inline(__always)
    func all(using deviceStream: DeviceStream? = nil) throws -> Self {
        
        var result = Self.init(shapedLike: self)
        try Netlib.all(self, result: &result, using: deviceStream)
        return result
    }
}
