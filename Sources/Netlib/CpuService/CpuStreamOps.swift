//******************************************************************************
//  Created by Edward Connell on 3/29/19
//  Copyright © 2019 Connell Research. All rights reserved.
//
import Foundation
import TensorFlow


extension CpuStream {
    public func abs<T>(x: T, result: inout T) throws where T : TensorDataView, T.Scalar : SignedNumeric {
        
    }
    
    public func add<TL, TR>(lhs: TL, rhs: TR, result: inout TL) throws where TL : TensorDataView, TR : TensorDataView, TL.Scalar : Numeric, TR.Scalar : Numeric {
        
    }

    public func all<T>(x: T, result: inout T) throws where T : TensorDataView, T.Scalar == Bool {
        
    }
    
    public func any<T>(x: T, result: inout T) throws where T : TensorDataView, T.Scalar == Bool {
        
    }
    
    public func approximatelyEqual<T, R>(lhs: T, rhs: T, tolerance: Double, result: inout R) throws where T : TensorDataView, R : TensorDataView, T.Scalar : FloatingPoint, R.Scalar == Bool {
        
    }
    
    public func argmax<T, R>(x: T, squeezingAxis axis: Int, result: inout R) throws where T : TensorDataView, R : TensorDataView, T.Scalar : Numeric, R.Scalar == Int32 {
        
    }
    
    public func argmin<T, R>(x: T, squeezingAxis axis: Int, result: inout R) throws where T : TensorDataView, R : TensorDataView, T.Scalar : Numeric, R.Scalar == Int32 {
        
    }
    
    public func broadcast<T>(x: T, toShape shape: DataShape, result: inout T) throws where T : TensorDataView {
        
    }
    
    public func cast<T, R>(from: T, to result: inout R) throws where T : TensorDataView, R : TensorDataView, T.Scalar : AnyNumeric, R.Scalar : AnyNumeric {
        
    }
    
    public func ceil<T>(x: T, result: inout T) throws where T : TensorDataView, T.Scalar : FloatingPoint {
        
    }
    
    public func concatenate<T>(view: T, with other: T, alongAxis axis: Int, result: inout T) throws where T : TensorDataView {
        
    }
    
    public func cos<T>(x: T, result: inout T) throws where T : TensorDataView, T.Scalar : FloatingPoint {
        
    }
    
    public func cosh<T>(x: T, result: inout T) throws where T : TensorDataView, T.Scalar : FloatingPoint {
        
    }
    
    public func div<T>(lhs: T, rhs: T, result: inout T) throws where T : TensorDataView, T.Scalar : Numeric {
        
    }
    
    public func equal<T, R>(lhs: T, rhs: T, result: inout R) throws where T : TensorDataView, R : TensorDataView, T.Scalar : Numeric, R.Scalar == Bool {
        
    }
    
    public func exp<T>(x: T, result: inout T) throws where T : TensorDataView, T.Scalar : FloatingPoint {
        
    }
    
    public func floor<T>(x: T, result: inout T) throws where T : TensorDataView, T.Scalar : FloatingPoint {
        
    }
    
    public func greater<T, R>(lhs: T, rhs: T, result: inout R) throws where T : TensorDataView, R : TensorDataView, T.Scalar : Comparable, T.Scalar : Numeric, R.Scalar == Bool {
        
    }
    
    public func greaterOrEqual<T, R>(lhs: T, rhs: T, result: inout R) throws where T : TensorDataView, R : TensorDataView, T.Scalar : Comparable, T.Scalar : Numeric, R.Scalar == Bool {
        
    }
    
    public func less<T, R>(lhs: T, rhs: T, result: inout R) throws where T : TensorDataView, R : TensorDataView, T.Scalar : Comparable, T.Scalar : Numeric, R.Scalar == Bool {
        
    }
    
    public func lessOrEqual<T, R>(lhs: T, rhs: T, result: inout R) throws where T : TensorDataView, R : TensorDataView, T.Scalar : Comparable, T.Scalar : Numeric, R.Scalar == Bool {
        
    }
    
    public func log<T>(x: T, result: inout T) throws where T : TensorDataView, T.Scalar : FloatingPoint {
        
    }
    
    public func logicalNot<T>(x: T, result: inout T) throws where T : TensorDataView, T.Scalar == Bool {
        
    }
    
    public func logicalAnd<T>(lhs: T, rhs: T, result: inout T) throws where T : TensorDataView, T.Scalar == Bool {
        
    }
    
    public func logicalOr<T>(lhs: T, rhs: T, result: inout T) throws where T : TensorDataView, T.Scalar == Bool {
        
    }
    
    public func logSoftmax<T>(x: T, result: inout T) throws where T : TensorDataView, T.Scalar : FloatingPoint {
        
    }
    
    public func matmul<T>(lhs: T, rhs: T, result: inout T) throws where T : TensorDataView, T.Scalar : Numeric {
        
    }
    
    public func max<T>(x: T, squeezingAxes axes: [Int], result: inout T) throws where T : TensorDataView, T.Scalar : Numeric {
        
    }
    
    public func maximum<T>(lhs: T, rhs: T, result: inout T) throws where T : TensorDataView, T.Scalar : Numeric {
        
    }
    
    public func mean<T>(x: T, squeezingAxes axes: [Int], result: inout T) throws where T : TensorDataView, T.Scalar : Numeric {
        
    }
    
    public func min<T>(x: T, squeezingAxes axes: [Int], result: inout T) throws where T : TensorDataView, T.Scalar : Numeric {
        
    }
    
    public func minimum<T>(lhs: T, rhs: T, result: inout T) throws where T : TensorDataView, T.Scalar : Numeric {
        
    }
    
    public func mod<T>(lhs: T, rhs: T, result: inout T) throws where T : TensorDataView, T.Scalar : Numeric {
        
    }
    
    public func mul<T>(lhs: T, rhs: T, result: inout T) throws where T : TensorDataView, T.Scalar : Numeric {
        
    }
    
    public func neg<T>(x: T, result: inout T) throws where T : TensorDataView, T.Scalar: Numeric {
        
    }
    
    public func notEqual<T, R>(lhs: T, rhs: T, result: inout R) throws where T : TensorDataView, R : TensorDataView, T.Scalar : Comparable, T.Scalar : Numeric, R.Scalar == Bool {
        
    }
    
    public func pad<T>(x: T, with margins: [(before: Int, after: Int)], fillValue: T.Scalar, result: inout T) throws where T : TensorDataView {
        
    }
    
    public func pow<TX, TY, TR>(x: TX, y: TY, result: inout TR) throws where TX : TensorDataView, TY : TensorDataView, TR : TensorDataView, TX.Scalar : Numeric, TY.Scalar : Numeric, TR.Scalar : Numeric {
        
    }

    public func prod<T>(x: T, result: inout T) throws where T : TensorDataView, T.Scalar : Numeric {
        
    }
    
    public func rsqrt<T>(x: T, result: inout T) throws where T : TensorDataView, T.Scalar : FloatingPoint {
        
    }
    
    public func replacing<T, R>(x: T, with other: T, where mask: R, result: inout T) throws where T : TensorDataView, R : TensorDataView, R.Scalar == Bool {
        
    }
    
    public func sin<T>(x: T, result: inout T) throws where T : TensorDataView, T.Scalar : FloatingPoint {
        
    }
    
    public func sinh<T>(x: T, result: inout T) throws where T : TensorDataView, T.Scalar : FloatingPoint {
        
    }
    
    public func square<T>(x: T, result: inout T) throws where T : TensorDataView, T.Scalar : Numeric {
        
    }
    
    public func squaredDifference<T>(lhs: T, rhs: T, result: inout T) throws where T : TensorDataView, T.Scalar : Numeric {
        
    }
    
    public func sqrt<T>(x: T, result: inout T) throws where T : TensorDataView, T.Scalar : FloatingPoint {
        
    }
    
    public func subtract<T>(lhs: T, rhs: T, result: inout T) throws where T : TensorDataView, T.Scalar : Numeric {
        
    }
    
    public func sum<T>(x: T, result: inout T) throws where T : TensorDataView, T.Scalar : Numeric {
        
    }
    
    public func tan<T>(x: T, result: inout T) throws where T : TensorDataView, T.Scalar : FloatingPoint {
        
    }
    
    public func tanh<T>(x: T, result: inout T) throws where T : TensorDataView, T.Scalar : FloatingPoint {
        
    }
}
