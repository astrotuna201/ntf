//******************************************************************************
//  Created by Edward Connell on 3/24/19
//  Copyright © 2019 Connell Research. All rights reserved.
//
import Foundation
import TensorFlow

//===----------------------------------------------------------------------===//
// Broadcasting
//===----------------------------------------------------------------------===//

public extension TensorView {
    @inlinable @inline(__always)
    func broadcast(toShape shape: Shape) -> TensorView {
        fatalError()
//        return Raw.broadcastTo(self, shape: shape)
    }

    /// Broadcast to the same shape as the specified `TensorView`.
    /// - Precondition: The specified shape must be compatible for broadcasting.
    @inlinable @inline(__always)
    func broadcast<OtherScalar>(like other: TensorView<OtherScalar>) -> TensorView {
        return broadcast(toShape: other.shape)
    }
}

//==============================================================================
// Reduction
extension TensorView where Scalar : TensorFlowFloatingPoint {
    @inlinable
    func _vjpMean() -> (TensorView, (TensorView) -> TensorView) {
        return (mean(), { [shape = shape, count = shape.elementCount] in
            ($0 / TensorView(count)).broadcast(toShape: shape)
        })
    }
    
    @inlinable
    func _vjpSum() -> (TensorView, (TensorView) -> TensorView) {
        return (sum(), { [shape = self.shape] in $0.broadcast(toShape: shape) })
    }
    
//    @inlinable
//    func _vjpMean(alongAxes axes: [Int]) -> (TensorView, (TensorView) -> TensorView) {
//        let value = mean(alongAxes: axes)
//        return (value, { [shape = shapeTensor, count = scalarCountTensor] in
//            $0.broadcast(toShape: shape) / TensorView(count)
//        })
//    }
//
//    @inlinable
//    func _vjpSum(alongAxes axes: [Int]) -> (TensorView, (TensorView) -> TensorView) {
//        let value = sum(alongAxes: axes)
//        return (value, { [shape = shapeTensor] in $0.broadcast(toShape: shape) })
//    }
}

public extension TensorView where Scalar: AnyNumber {
    
    /// Returns the indices of the maximum values along the specified axes. The
    /// reduced dimensions are removed.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable @inline(__always)
    func argmax(squeezingAxis axis: Int) -> TensorView<Int32> {
        fatalError()
        //return Raw.argMax(self, dimension: TensorView<Int32>(axis))
    }
    
    // NOTE: This overload is necessary, otherwise `sum()` would refer
    // to the variadic method `sum(squeezingAxes:)` with zero indices.
    @inlinable @inline(__always)
    //    @differentiable(
    //    wrt: self, vjp: _vjpSum()
    //    where Scalar : TensorFlowFloatingPoint
    //    )
    func sum() -> TensorView {
        fatalError()
    }
    
    // NOTE: This overload is necessary, otherwise `mean()` would refer
    // to the variadic method `mean(squeezingAxes:)` with zero indices.
    @differentiable(
    wrt: self, vjp: _vjpMean()
    where Scalar : AnyNumber & TensorFlowFloatingPoint
    )
    @inlinable @inline(__always)
    func mean() -> TensorView {
        fatalError()
        //        let axes = TensorView<Int32>(rangeFrom: 0, to: rank, stride: 1)
        //        return Raw.mean(self, reductionIndices: axes)
    }
    
}
