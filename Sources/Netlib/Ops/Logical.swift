//******************************************************************************
// Created by Edward Connell on 4/3/19
// Copyright © 2019 Connell Research. All rights reserved.
//
import Foundation

//==============================================================================
/// approximatelyEqual(lhs:rhs:
///
/// Performs an element wise comparison of two tensors within the specified
/// `tolerance`
infix operator ≈ : ComparisonPrecedence

/// in place
/// - Parameter lhs: left hand tensor
/// - Parameter rhs: right hand tensor
/// - Parameter result: the tensor where the result will be written
@inlinable @inline(__always)
public func approximatelyEqual<T>(
    _ lhs: T, _ rhs: T, result: inout T.BoolView, tolerance: Double = 0.00001,
    using deviceStream: DeviceStream? = nil) throws
    where T: TensorView, T.Scalar: FloatingPoint & AnyConvertable {
        
        let toleranceTensor = ScalarTensor(T.Scalar(any: tolerance))
        let stream = deviceStream ?? _ThreadLocal.value.defaultStream
        try stream.approximatelyEqual(lhs: lhs, rhs: rhs,
                                      tolerance: toleranceTensor,
                                      result: &result)
}

/// returns new view
/// - Parameter rhs: right hand tensor
/// - Returns: a new tensor containing the result
public extension TensorView where Self.Scalar: FloatingPoint & AnyConvertable {
    @inlinable @inline(__always)
    func approximatelyEqual(_ rhs: Self, tolerance: Double = 0.00001,
                            using deviceStream: DeviceStream? = nil) throws
        -> Self.BoolView {
            var result = Self.BoolView.init(shapedLike: self)
            try Netlib.approximatelyEqual(self, rhs, result: &result,
                                          tolerance: tolerance,
                                          using: deviceStream)
            return result
    }
}
