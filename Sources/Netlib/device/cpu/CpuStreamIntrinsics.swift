//******************************************************************************
//  Created by Edward Connell on 4/16/19
//  Copyright © 2019 Connell Research. All rights reserved.
//
import Foundation


///****** THESE NEED TO BE SMOOTHER!!
///       work is in flux

public extension CpuStream {
    func abs<T>(x: T, result: inout T) where
        T: TensorView, T.Scalar: SignedNumeric, T.Scalar.Magnitude == T.Scalar
    {
        
    }
    
    //--------------------------------------------------------------------------
    /// add
    func add<T>(lhs: T, rhs: T, result: inout T) where
        T : TensorView, T.Scalar : Numeric
    {
        guard lastError == nil else { return }
        do {
            var resultRef = try result.reference(using: self)
            var results = try resultRef.mutableDeviceValues(using: self)
            let lhs = try lhs.deviceValues(using: self)
            let rhs = try rhs.deviceValues(using: self)
            queue {
                zip(lhs, rhs).map(to: &results) { $0 + $1 }
            }
        } catch {
            reportDevice(error: error, event: completionEvent)
        }
    }
    
    //--------------------------------------------------------------------------
    /// all
    func all<T>(x: T, axes: Vector<IndexScalar>?, result: inout T) where
        T : TensorView, T.Scalar == Bool
    {
        guard lastError == nil else { return }
        do {
            var resultRef = try result.reference(using: self)
            let result = try resultRef.readWrite(using: self)
            let x = try x.deviceValues(using: self)
            queue {
                for value in x where !value {
                    result[0] = false
                    return
                }
                result[0] = true
            }
        } catch {
            reportDevice(error: error, event: completionEvent)
        }
    }
    
    //--------------------------------------------------------------------------
    /// any
    func any<T>(x: T, axes: Vector<IndexScalar>?, result: inout T) where
        T : TensorView, T.Scalar == Bool
    {
        guard lastError == nil else { return }
        do {
            var resultRef = try result.reference(using: self)
            let result = try resultRef.readWrite(using: self)
            let x = try x.deviceValues(using: self)
            queue {
                for value in x where value {
                    result[0] = true
                    return
                }
                result[0] = false
            }
        } catch {
            reportDevice(error: error, event: completionEvent)
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
        guard lastError == nil else { return }
        do {
            var resultRef = try result.reference(using: self)
            var results = try resultRef.mutableDeviceValues(using: self)
            let lhs = try lhs.deviceValues(using: self)
            let rhs = try rhs.deviceValues(using: self)
            queue {
                zip(lhs, rhs).map(to: &results) { $0.0 - $0.1 <= tolerance }
            }
        } catch {
            reportDevice(error: error, event: completionEvent)
        }
    }
    
    func argmax<T>(x: T, axes: Vector<IndexScalar>?, result: inout T.IndexView) where T : TensorView, T.Scalar : Numeric, T.IndexView.Scalar == IndexScalar {
        
    }
    
    func argmin<T>(x: T, axes: Vector<IndexScalar>?, result: inout T.IndexView) where T : TensorView, T.Scalar : Numeric, T.IndexView.Scalar == IndexScalar {
        
    }
    
    //--------------------------------------------------------------------------
    /// asum
    func asum<T>(x: T, axes: Vector<IndexScalar>?, result: inout T) where
        T: TensorView, T.Scalar: SignedNumeric, T.Scalar.Magnitude == T.Scalar
    {
        guard lastError == nil else { return }
        do {
            var resultRef = try result.reference(using: self)
            var results = try resultRef.mutableDeviceValues(using: self)
            let x = try x.deviceValues(using: self)
            queue {
                x.reduce(to: &results, T.Scalar.zero) {
                    $0 + $1.magnitude
                }
            }
        } catch {
            reportDevice(error: error, event: completionEvent)
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
    
    //--------------------------------------------------------------------------
    /// div
    func div<T>(lhs: T, rhs: T, result: inout T) where
        T : TensorView, T.Scalar : FloatingPoint
    {
        guard lastError == nil else { return }
        do {
            var resultRef = try result.reference(using: self)
            var results = try resultRef.mutableDeviceValues(using: self)
            let lhs = try lhs.deviceValues(using: self)
            let rhs = try rhs.deviceValues(using: self)
            queue {
                zip(lhs, rhs).map(to: &results) { $0 / $1 }
            }
        } catch {
            reportDevice(error: error, event: completionEvent)
        }
    }
    
    //--------------------------------------------------------------------------
    /// equal
    func equal<T>(lhs: T, rhs: T, result: inout T.BoolView) where
        T: TensorView, T.Scalar: Equatable,
        T.BoolView.Scalar == Bool
    {
        guard lastError == nil else { return }
        do {
            var resultRef = try result.reference(using: self)
            var results = try resultRef.mutableDeviceValues(using: self)
            let lhs = try lhs.deviceValues(using: self)
            let rhs = try rhs.deviceValues(using: self)
            queue {
                zip(lhs, rhs).map(to: &results) { $0 == $1 }
            }
        } catch {
            reportDevice(error: error, event: completionEvent)
        }
    }
    
    //--------------------------------------------------------------------------
    /// exp
    func exp<T>(x: T, result: inout T) where
        T : TensorView, T.Scalar : FloatingPoint
    {
        
    }
    
    //--------------------------------------------------------------------------
    /// fill(result:with:
    func fill<T>(_ result: inout T, with value: T.Scalar) where T : TensorView {
        guard lastError == nil else { return }
        do {
            var resultRef = try result.reference(using: self)
            var values = try resultRef.mutableDeviceValues(using: self)
            queue {
                for index in values.indices {
                    values[index] = value
                }
            }
        } catch {
            reportDevice(error: error, event: completionEvent)
        }
    }
    
    //--------------------------------------------------------------------------
    /// fillWithIndex(x:startAt:
    func fillWithIndex<T>(_ result: inout T, startAt: Int) where
        T : TensorView, T.Scalar: AnyNumeric
    {
        guard lastError == nil else { return }
        do {
            var resultRef = try result.reference(using: self)
            var values = try resultRef.mutableDeviceValues(using: self)
            queue {
                var value = startAt
                for index in values.indices {
                    values[index] = T.Scalar(any: value)
                    value += 1
                }
            }
        } catch {
            reportDevice(error: error, event: completionEvent)
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

    //--------------------------------------------------------------------------
    /// log(x:result:
    func log<T>(x: T, result: inout T) where
        T: TensorView, T.Scalar: AnyFloatingPoint
    {
        guard lastError == nil else { return }
        do {
            var resultRef = try result.reference(using: self)
            var results = try resultRef.mutableDeviceValues(using: self)
            let x = try x.deviceValues(using: self)
            queue {
                x.map(to: &results) {
                    T.Scalar(any: Foundation.log($0.asFloat))
                }
            }
        } catch {
            reportDevice(error: error, event: completionEvent)
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
        guard lastError == nil else { return }
        do {
            var resultRef = try result.reference(using: self)
            var results = try resultRef.mutableDeviceValues(using: self)
            let lhs = try lhs.deviceValues(using: self)
            let rhs = try rhs.deviceValues(using: self)
            queue {
                zip(lhs, rhs).map(to: &results) { $0 * $1 }
            }
        } catch {
            reportDevice(error: error, event: completionEvent)
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
        T : TensorView, T.Scalar : AnyNumeric
    {
        guard lastError == nil else { return }
        do {
            var resultRef = try result.reference(using: self)
            var results = try resultRef.mutableDeviceValues(using: self)
            let x = try x.deviceValues(using: self)
            let y = try y.deviceValues(using: self)
            queue {
                zip(x, y).map(to: &results) {
                    T.Scalar(any: Foundation.pow($0.asDouble, $1.asDouble))
                }
            }
        } catch {
            reportDevice(error: error, event: completionEvent)
        }
    }

    //--------------------------------------------------------------------------
    // prod
    func prod<T>(x: T, axes: Vector<IndexScalar>?, result: inout T) where
        T : TensorView, T.Scalar : AnyNumeric
    {
        guard lastError == nil else { return }
        do {
            var resultRef = try result.reference(using: self)
            var results = try resultRef.mutableDeviceValues(using: self)
            let x = try x.deviceValues(using: self)
            queue {
                x.reduce(to: &results, T.Scalar(any: 1)) { $0 * $1 }
            }
        } catch {
            reportDevice(error: error, event: completionEvent)
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
    
    //--------------------------------------------------------------------------
    // subtract
    func subtract<T>(lhs: T, rhs: T, result: inout T) where
        T : TensorView, T.Scalar : Numeric
    {
        guard lastError == nil else { return }
        do {
            var resultRef = try result.reference(using: self)
            var results = try resultRef.mutableDeviceValues(using: self)
            let lhs = try lhs.deviceValues(using: self)
            let rhs = try rhs.deviceValues(using: self)
            queue {
                zip(lhs, rhs).map(to: &results) { $0 - $1 }
            }
        } catch {
            reportDevice(error: error, event: completionEvent)
        }
    }
    
    //--------------------------------------------------------------------------
    // sum
    func sum<T>(x: T, axes: Vector<IndexScalar>?, result: inout T) where
        T : TensorView, T.Scalar : Numeric
    {
        guard lastError == nil else { return }
        do {
            var resultRef = try result.reference(using: self)
            var results = try resultRef.mutableDeviceValues(using: self)
            let xseq = try x.deviceValues(using: self)
            queue {
                xseq.reduce(to: &results, T.Scalar.zero) { $0 + $1 }
            }
        } catch {
            reportDevice(error: error, event: completionEvent)
        }
    }
    
    func tan<T>(x: T, result: inout T) where T : TensorView, T.Scalar : FloatingPoint {
        
    }
    
    func tanh<T>(x: T, result: inout T) where T : TensorView, T.Scalar : FloatingPoint {
        
    }
}
