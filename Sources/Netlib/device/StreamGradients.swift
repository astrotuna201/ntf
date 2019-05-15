//******************************************************************************
//  Created by Edward Connell on 4/16/19
//  Copyright © 2019 Connell Research. All rights reserved.
//
import Foundation

//==============================================================================
// StreamGradientsProtocol
/// The required set of base level intrinsic gradient functions for a
/// `DeviceStream`. This is a remoting interface so no default implementation
/// is provided (see StreamGradients)
public protocol StreamGradientsProtocol {
    /// _vjpAdd
    static func vjpAdd<T>(_ lhs: T, _ rhs: T, result: inout T) ->
        (T, (T) -> (T, T)) where T: TensorView, T.Element: FloatingPoint
    /// _vjpSubtract
    static func vjpSubtract<T>(_ lhs: T, _ rhs: T, result: inout T) ->
        (T, (T) -> (T, T)) where T: TensorView, T.Element: FloatingPoint
    /// _vjpMultiply
    static func vjpMultiply<T>(_ lhs: T, _ rhs: T, result: inout T) ->
        (T, (T) -> (T, T)) where T: TensorView, T.Element: FloatingPoint
    /// _vjpDivide
    static func vjpDivide<T>(_ lhs: T, _ rhs: T, result: inout T) ->
        (T, (T) -> (T, T)) where T: TensorView, T.Element: FloatingPoint
    
    /// And all the rest........
}

//==============================================================================
/// StreamGradients
/// `DeviceStream` implementations can adopt StreamGradients to pick up a
/// default implementation based on calls to the required implementation of
/// the `StreamIntrinsicsProtocol`
/// However, stream implementations can specialize and offer high performance
/// device specific gradient function implementations.
public protocol StreamGradients: StreamGradientsProtocol { }

//==============================================================================
/// StreamGradients

//*** TODO: we do broadcasting differently and unbroadcast is a REALLY
//*** expensive function, so rethink this

extension StreamGradients where Self: StreamIntrinsicsProtocol {
    @inlinable
    public static func vjpAdd<T>(_ lhs: T, _ rhs: T, result: inout T) ->
        (T, (T) -> (T, T)) where T: TensorView, T.Element: FloatingPoint {
        fatalError("not implemented yet")
//        return (lhs + rhs, {
//            [lhsShape = lhs.shape, rhsShape = rhs.shape] v in
//            return (v.unbroadcast(toShape: lhsShape),
//                    v.unbroadcast(toShape: rhsShape))
//        })
    }
    
    @inlinable
    public static func vjpSubtract<T>(_ lhs: T, _ rhs: T, result: inout T) ->
        (T, (T) -> (T, T)) where T: TensorView, T.Element: FloatingPoint {
        fatalError("not implemented yet")
//        return (lhs - rhs, {
//            [lhsShape = lhs.shape, rhsShape = rhs.shape] v in
//            return (v.unbroadcast(toShape: lhsShape),
//                    -v.unbroadcast(toShape: rhsShape))
//        })
    }

    @inlinable
    public static func vjpMultiply<T>(_ lhs: T, _ rhs: T, result: inout T) ->
        (T, (T) -> (T, T)) where T: TensorView, T.Element: FloatingPoint {
        fatalError("not implemented yet")
//        return (lhs * rhs, {
//            [lhsShape = lhs.shape, rhsShape = rhs.shape] v in
//            ((rhs * v).unbroadcast(toShape: lhsShape),
//             (lhs * v).unbroadcast(toShape: rhsShape))
//        })
    }

    @inlinable
    public static func vjpDivide<T>(_ lhs: T, _ rhs: T, result: inout T) ->
        (T, (T) -> (T, T)) where T: TensorView, T.Element: FloatingPoint {
        fatalError("not implemented yet")
//        return (lhs / rhs, {
//            [lhsShape = lhs.shape, rhsShape = rhs.shape] v in
//            ((v / rhs).unbroadcast(toShape: lhsShape),
//             ((-lhs) / rhs.squared() * v).unbroadcast(toShape: rhsShape))
//        })
    }
}
