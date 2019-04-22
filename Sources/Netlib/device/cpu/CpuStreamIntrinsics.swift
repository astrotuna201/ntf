//******************************************************************************
//  Created by Edward Connell on 4/16/19
//  Copyright © 2019 Connell Research. All rights reserved.
//
import Foundation


///****** THESE NEED TO BE SMOOTHER!!
///       work is in flux

public extension CpuStream {
    func abs<T>(x: T, result: inout T) where T : TensorView, T.Scalar : SignedNumeric {
        
    }
    
    func add<T>(lhs: T, rhs: T, result: inout T) where T : TensorView, T.Scalar : Numeric {
        var resultRef = tryCatch { try result.reference() }
        queue {
            zip(lhs, rhs).map(to: &resultRef) { $0 + $1 }
        }
    }
    
    //--------------------------------------------------------------------------
    /// all
    func all<T>(x: T, axes: Vector<IndexScalar>?, result: inout T) where T : TensorView, T.Scalar == Bool {
        var resultRef = tryCatch { try result.reference() }
        queue {
            let values = try x.values()
            let buffer = try resultRef.readWrite()
            for value in values {
                if !value {
                    buffer[0] = false
                    return
                }
            }
            buffer[0] = true
        }
    }
    
    //--------------------------------------------------------------------------
    /// any
    func any<T>(x: T, axes: Vector<IndexScalar>?, result: inout T) where T : TensorView, T.Scalar == Bool {
        var resultRef = tryCatch { try result.reference() }
        queue {
            let values = try x.values()
            let buffer = try resultRef.readWrite()
            for value in values {
                if value {
                    buffer[0] = true
                    return
                }
            }
            buffer[0] = false
        }
    }
    
    //--------------------------------------------------------------------------
    /// approximatelyEqual
    func approximatelyEqual<T>(lhs: T, rhs: T,
                               tolerance: T.Scalar,
                               result: inout T.BoolView) where
        T : TensorView, T.Scalar : AnyFloatingPoint,
        T.BoolView.Scalar == Bool
    {
        var resultRef = tryCatch { try result.reference() }
        queue {
            zip(lhs, rhs).map(to: &resultRef) { $0.0 - $0.1 <= tolerance }
        }
    }
    
    func argmax<T>(x: T, axes: Vector<IndexScalar>?, result: inout T.IndexView) where T : TensorView, T.Scalar : Numeric, T.IndexView.Scalar == IndexScalar {
        
    }
    
    func argmin<T>(x: T, axes: Vector<IndexScalar>?, result: inout T.IndexView) where T : TensorView, T.Scalar : Numeric, T.IndexView.Scalar == IndexScalar {
        
    }
    
    func asum<T>(x: T, axes: Vector<IndexScalar>?, result: inout T)
        where T : TensorView, T.Scalar : AnyNumeric {
        var resultRef = tryCatch { try result.reference() }
        queue {
            try x.values().reduce(to: &resultRef, T.Scalar.zero) {
                // TODO: can't seem to call Foundation.abs($1)
                $0 + $1
            }
        }
    }
    
    func cast<T, R>(from: T, to result: inout R) where T : TensorView, R : TensorView, T.Scalar : AnyConvertable, R.Scalar : AnyConvertable {
        
    }
    
    func ceil<T>(x: T, result: inout T) where T : TensorView, T.Scalar : FloatingPoint {
        
    }
    
    func concatenate<T>(view: T, with other: T, alongAxis axis: Int, result: inout T) where T : TensorView {
        
    }
    
    func cos<T>(x: T, result: inout T) where T : TensorView, T.Scalar : FloatingPoint {
        
    }
    
    func cosh<T>(x: T, result: inout T) where T : TensorView, T.Scalar : FloatingPoint {
        
    }
    
    func div<T>(lhs: T, rhs: T, result: inout T) where T : TensorView, T.Scalar : FloatingPoint {
        var resultRef = tryCatch { try result.reference() }
        queue {
            zip(lhs, rhs).map(to: &resultRef) { $0 / $1 }
        }
    }
    
    func equal<T>(lhs: T, rhs: T, result: inout T.BoolView) where T : TensorView {
        
    }
    
    func exp<T>(x: T, result: inout T) where T : TensorView, T.Scalar : FloatingPoint {
        
    }
    
    //--------------------------------------------------------------------------
    /// fill(x:with:
    func fill<T>(x: inout T, with value: T.Scalar) where T : TensorView {
        var xref = tryCatch { try x.reference() }
        queue {
            var values = try xref.mutableDeviceValues(using: self)
            for i in values.startIndex..<values.endIndex {
                values[i] = value
            }
        }
    }
    
    //--------------------------------------------------------------------------
    /// fillWithIndex(x:startAt:
    func fillWithIndex<T>(x: inout T, startAt: Int) where
        T : TensorView, T.Scalar: AnyNumeric
    {
        var xref = tryCatch { try x.reference() }
        queue {
            var values = try xref.mutableDeviceValues(using: self)
            var value = startAt
            for i in values.startIndex..<values.endIndex {
                values[i] = T.Scalar(any: value)
                value += 1
            }
        }
    }

    func floor<T>(x: T, result: inout T) where T : TensorView, T.Scalar : FloatingPoint {
        
    }
    
    func greater<T>(lhs: T, rhs: T, result: inout T.BoolView) where T : TensorView, T.Scalar : Numeric {
        
    }
    
    func greaterOrEqual<T>(lhs: T, rhs: T, result: inout T.BoolView) where T : TensorView, T.Scalar : Numeric {
        
    }
    
    func less<T>(lhs: T, rhs: T, result: inout T.BoolView) where T : TensorView, T.Scalar : Numeric {
        
    }
    
    func lessOrEqual<T>(lhs: T, rhs: T, result: inout T.BoolView) where T : TensorView, T.Scalar : Numeric {
        
    }
    
    func log<T>(x: T, result: inout T) where T : TensorView, T.Scalar : AnyFloatingPoint {
        var resultRef = tryCatch { try result.reference() }
        queue { try x.values().map(to: &resultRef) {
            T.Scalar(any: Foundation.log($0.asDouble)) }
        }
    }
    
    func logicalNot<T>(x: T, result: inout T) where T : TensorView, T.Scalar == Bool {
        
    }
    
    func logicalAnd<T>(lhs: T, rhs: T, result: inout T) where T : TensorView, T.Scalar == Bool {
        
    }
    
    func logicalOr<T>(lhs: T, rhs: T, result: inout T) where T : TensorView, T.Scalar == Bool {
        
    }
    
    func logSoftmax<T>(x: T, result: inout T) where T : TensorView, T.Scalar : FloatingPoint {
        
    }
    
    func matmul<T>(lhs: T, rhs: T, result: inout T) where T : TensorView, T.Scalar : Numeric {
        
    }
    
    func max<T>(x: T, squeezingAxes axes: [Int], result: inout T) where T : TensorView, T.Scalar : Numeric {
        
    }
    
    func maximum<T>(lhs: T, rhs: T, result: inout T) where T : TensorView, T.Scalar : Numeric {
        
    }
    
    func mean<T>(x: T, axes: Vector<IndexScalar>?, result: inout T) where T : TensorView, T.Scalar : Numeric {
        
    }
    
    func min<T>(x: T, squeezingAxes axes: [Int], result: inout T) where T : TensorView, T.Scalar : Numeric {
        
    }
    
    func minimum<T>(lhs: T, rhs: T, result: inout T) where T : TensorView, T.Scalar : Numeric {
        
    }
    
    func mod<T>(lhs: T, rhs: T, result: inout T) where T : TensorView, T.Scalar : Numeric {
        
    }
    
    func mul<T>(lhs: T, rhs: T, result: inout T) where T : TensorView, T.Scalar : Numeric {
        var resultRef = tryCatch { try result.reference() }
        queue {
            zip(lhs, rhs).map(to: &resultRef) { $0 * $1 }
        }
    }
    
    func neg<T>(x: T, result: inout T) where T : TensorView, T.Scalar : SignedNumeric {
        
    }
    
    func notEqual<T>(lhs: T, rhs: T, result: inout T.BoolView) where T : TensorView, T.Scalar : Numeric {
        
    }

    //--------------------------------------------------------------------------
    // pow
    // TODO something is wrong, I shouldn't need to do this to interface
    // with math functions
    func pow<T>(x: T, y: T, result: inout T) where
        T : TensorView, T.Scalar == Float {
            
        var resultRef = tryCatch { try result.reference() }
        queue { zip(x, y).map(to: &resultRef) { powf($0, $1) } }
    }

    func pow<T>(x: T, y: T, result: inout T) where
        T : TensorView, T.Scalar : AnyFloatingPoint {
            
        var resultRef = tryCatch { try result.reference() }
        queue {
            zip(x, y).map(to: &resultRef) {
                T.Scalar(any: Foundation.pow($0.asDouble, $1.asDouble))
            }
        }
    }

    //--------------------------------------------------------------------------
    // prod
    func prod<T>(x: T, axes: Vector<IndexScalar>?, result: inout T) where
        T : TensorView, T.Scalar : Numeric {
        var resultRef = tryCatch { try result.reference() }
        queue {
            try x.values().reduce(to: &resultRef, T.Scalar.zero) { $0 * $1 }
        }
    }
    
    func rsqrt<T>(x: T, result: inout T) where T : TensorView, T.Scalar : FloatingPoint {
        
    }
    
    func replacing<T>(x: T, with other: T, where mask: T.BoolView, result: inout T) where T : TensorView {
        
    }
    
    func sin<T>(x: T, result: inout T) where T : TensorView, T.Scalar : FloatingPoint {
        
    }
    
    func sinh<T>(x: T, result: inout T) where T : TensorView, T.Scalar : FloatingPoint {
        
    }
    
    func square<T>(x: T, result: inout T) where T : TensorView, T.Scalar : Numeric {
        
    }
    
    func squaredDifference<T>(lhs: T, rhs: T, result: inout T) where T : TensorView, T.Scalar : Numeric {
        
    }
    
    func sqrt<T>(x: T, result: inout T) where T : TensorView, T.Scalar : FloatingPoint {
        
    }
    
    func subtract<T>(lhs: T, rhs: T, result: inout T) where T : TensorView, T.Scalar : Numeric {
        var resultRef = tryCatch { try result.reference() }
        queue {
            zip(lhs, rhs).map(to: &resultRef) { $0 - $1 }
        }
    }
    
    func sum<T>(x: T, axes: Vector<IndexScalar>?, result: inout T) where
        T : TensorView, T.Scalar : Numeric {
        var resultRef = tryCatch { try result.reference() }
        queue {
            try x.values().reduce(to: &resultRef, T.Scalar.zero) { $0 + $1 }
        }
    }
    
    func tan<T>(x: T, result: inout T) where T : TensorView, T.Scalar : FloatingPoint {
        
    }
    
    func tanh<T>(x: T, result: inout T) where T : TensorView, T.Scalar : FloatingPoint {
        
    }
}
