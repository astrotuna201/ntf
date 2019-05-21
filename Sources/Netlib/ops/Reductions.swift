//******************************************************************************
// Created by Edward Connell on 4/3/19
// Copyright © 2019 Connell Research. All rights reserved.
//
import Foundation

//==============================================================================
/// all(x:alongAxes:)
/// Returns `true` if all values are equal to `true` along the specified
/// axes. Otherwise returns `false`. The result extent along the specified
/// axes will be 1. Rank is not reduced.

/// in place
/// - Parameter x: value tensor
/// - Parameter alongAxes: the axes to operate on
/// - Parameter result: the scalar tensor where the result will be written
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
@inlinable @inline(__always)
public func all<T>(_ x: T, alongAxes axes: Vector<IndexElement>? = nil,
                   result: inout T)
    where T: TensorView, T.Element == Bool
{
    _Streams.current.all(x: x, along: axes, result: &result)
}

/// returns new view
/// - Parameter alongAxes: the axes to operate on
/// - Returns: a new tensor containing the result
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
public extension TensorView where Element == Bool {
    @inlinable @inline(__always)
    func all(alongAxes: Int...) -> Self {
        // turn into a vector
        let axes = Vector<IndexElement>(
            any: shape.makePositive(indices: alongAxes))
        var result = createDense()
        Netlib.all(self, alongAxes: axes, result: &result)
        return result
    }
    
    @inlinable @inline(__always)
    func all() -> Self {
        let extents = [Int](repeating: 1, count: shape.rank)
        var result = createDense(with: extents)
        Netlib.all(self, result: &result)
        return result
    }
    
    /// returns new view
    /// - Parameter alongAxes: the axes to operate on
    /// - Returns: a new NDTensor containing the result
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable @inline(__always)
    func all(squeezing: Int...) -> NDTensor<Element> {
        let axes = shape.makePositive(indices: squeezing)
        let axesVec = Vector<IndexElement>(any: axes)
        var result = createDense()
        Netlib.all(self, alongAxes: axesVec, result: &result)
        return result.squeezed(axes: axes)
    }
}

//==============================================================================
/// mean(x:alongAxes:)
/// Returns the mean of all values along the specified
/// axes. The result extent along the specified axes will be 1.
/// Rank is not reduced.

/// to result
/// - Parameter x: value tensor
/// - Parameter alongAxes: the axes to operate on
/// - Parameter result: the scalar tensor where the result will be written
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
@inlinable @inline(__always)
public func mean<T>(_ x: T, alongAxes axes: Vector<IndexElement>? = nil,
                    result: inout T)
    where T: TensorView, T.Element: FloatingPoint
{
    _Streams.current.mean(x: x, along: axes, result: &result)
}

/// return result
/// - Parameter x: value tensor
/// - Parameter alongAxes: the axes to operate on
/// - Parameter result: the scalar tensor where the result will be written
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
@inlinable @inline(__always)
public func mean<T>(_ x: T, alongAxes axes: Vector<IndexElement>? = nil) -> T
    where T: TensorView, T.Element: FloatingPoint
{
    let extents = [Int](repeating: 1, count: x.rank)
    var result = x.createDense(with: extents)
    _Streams.current.mean(x: x, along: axes, result: &result)
    return result
}

/// returns new view
/// - Parameter alongAxes: the axes to operate on
/// - Returns: a new tensor containing the result
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
public extension TensorView where Element: FloatingPoint {
    @inlinable @inline(__always)
    func mean(alongAxes axes: Vector<IndexElement>? = nil) -> Self {
        var result = createDense()
        Netlib.mean(self, alongAxes: axes, result: &result)
        return result
    }

    @inlinable @inline(__always)
    func mean(alongAxes: [Int]) -> Self {
        // turn into a vector
        let axes = Vector<IndexElement>(
            any: shape.makePositive(indices: alongAxes))
        return mean(alongAxes: axes)
    }

    @inlinable @inline(__always)
    func mean(alongAxes: Int...) -> Self {
        return mean(alongAxes: alongAxes)
    }
    
    @inlinable @inline(__always)
    func mean() -> Self {
        let extents = [Int](repeating: 1, count: shape.rank)
        var result = createDense(with: extents)
        Netlib.mean(self, result: &result)
        return result
    }
    
    /// returns new view
    /// - Parameter alongAxes: the axes to operate on
    /// - Returns: a new NDTensor containing the result
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable @inline(__always)
    func mean(squeezingAxes: Int...) -> NDTensor<Element> {
        let axes = shape.makePositive(indices: squeezingAxes)
        let axesVec = Vector<IndexElement>(any: axes)
        var result = createDense()
        Netlib.mean(self, alongAxes: axesVec, result: &result)
        return result.squeezed(axes: axes)
    }
}

//==============================================================================
/// sum(x:alongAxes:)
/// Returns the sum of all values are along the specified
/// axes. The result extent along the specified axes will be 1.
/// Rank is not reduced.

/// to result
/// - Parameter x: value tensor
/// - Parameter alongAxes: the axes to operate on
/// - Parameter result: the scalar tensor where the result will be written
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
@inlinable @inline(__always)
public func sum<T>(_ x: T, alongAxes axes: Vector<IndexElement>? = nil,
                   result: inout T)
    where T: TensorView, T.Element: Numeric
{
    _Streams.current.sum(x: x, along: axes, result: &result)
}

/// return result
/// - Parameter x: value tensor
/// - Parameter alongAxes: the axes to operate on
/// - Parameter result: the scalar tensor where the result will be written
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
@inlinable @inline(__always)
public func sum<T>(_ x: T, alongAxes axes: Vector<IndexElement>? = nil) -> T
    where T: TensorView, T.Element: Numeric
{
    let extents = [Int](repeating: 1, count: x.rank)
    var result = x.createDense(with: extents)
    _Streams.current.sum(x: x, along: axes, result: &result)
    return result
}

/// returns new view
/// - Parameter alongAxes: the axes to operate on
/// - Returns: a new tensor containing the result
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
public extension TensorView where Element: Numeric {
    @inlinable @inline(__always)
    func sum(alongAxes: Int...) -> Self {
        // turn into a vector
        let axes = Vector<IndexElement>(
            any: shape.makePositive(indices: alongAxes))
        var result = createDense()
        Netlib.sum(self, alongAxes: axes, result: &result)
        return result
    }
    
    @inlinable @inline(__always)
    func sum() -> Self {
        let extents = [Int](repeating: 1, count: shape.rank)
        var result = createDense(with: extents)
        Netlib.sum(self, result: &result)
        return result
    }
    
    /// returns new view
    /// - Parameter alongAxes: the axes to operate on
    /// - Returns: a new NDTensor containing the result
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable @inline(__always)
    func sum(squeezingAxes: Int...) -> NDTensor<Element> {
        let axes = shape.makePositive(indices: squeezingAxes)
        let axesVec = Vector<IndexElement>(any: axes)
        var result = createDense()
        Netlib.sum(self, alongAxes: axesVec, result: &result)
        return result.squeezed(axes: axes)
    }
}

//==============================================================================
/// variance(x:alongAxes:)
/// Returns the variance of all values are along the specified
/// axes. The result extent along the specified axes will be 1.
/// Rank is not reduced.

/// to result
/// - Parameter x: value tensor
/// - Parameter alongAxes: the axes to operate on
/// - Parameter result: the scalar tensor where the result will be written
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
@inlinable @inline(__always)
public func variance<T>(_ x: T, alongAxes axes: Vector<IndexElement>? = nil,
                        result: inout T)
    where T: TensorView, T.Element: FloatingPoint
{
    let sqdiff = squaredDifference(x, x.mean(alongAxes: axes))
    _Streams.current.mean(x: sqdiff, along: axes, result: &result)
}

/// to result
/// - Parameter x: value tensor
/// - Parameter alongAxes: the axes to operate on
/// - Parameter meanValue: the tensor where the mean will be written
/// - Parameter result: the scalar tensor where the result will be written
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
@inlinable @inline(__always)
public func variance<T>(_ x: T, alongAxes axes: Vector<IndexElement>? = nil,
                        meanValue: inout T, result: inout T)
    where T: TensorView, T.Element: FloatingPoint
{
    _Streams.current.mean(x: x, along: axes, result: &meanValue)
    let sqdiff = squaredDifference(x, meanValue)
    _Streams.current.mean(x: sqdiff, along: axes, result: &result)
}

/// return result
/// - Parameter x: value tensor
/// - Parameter alongAxes: the axes to operate on
/// - Parameter result: the scalar tensor where the result will be written
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
@inlinable @inline(__always)
public func variance<T>(_ x: T, alongAxes axes: Vector<IndexElement>? = nil) -> T
    where T: TensorView, T.Element: FloatingPoint
{
    var result = x.createDense(with: [Int](repeating: 1, count: x.rank))
    Netlib.variance(x, alongAxes: axes, result: &result)
    return result
}

/// returns new view
/// - Parameter alongAxes: the axes to operate on
/// - Returns: a new tensor containing the result
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
public extension TensorView where Element: FloatingPoint {
    @inlinable @inline(__always)
    func variance(alongAxes: [Int]) -> Self {
        // turn into a vector
        let axes = Vector<IndexElement>(
            any: shape.makePositive(indices: alongAxes))
        var result = createDense()
        Netlib.variance(self, alongAxes: axes, result: &result)
        return result
    }

    @inlinable @inline(__always)
    func variance(alongAxes: Int...) -> Self {
        return variance(alongAxes: alongAxes)
    }
    
    @inlinable @inline(__always)
    func variance() -> Self {
        return squaredDifference(self, self.mean()).mean()
    }
    
    /// returns new view
    /// - Parameter alongAxes: the axes to operate on
    /// - Returns: a new NDTensor containing the result
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable @inline(__always)
    func variance(squeezingAxes: Int...) -> NDTensor<Element> {
        let axes = shape.makePositive(indices: squeezingAxes)
        return variance(alongAxes: squeezingAxes).squeezed(axes: axes)
    }
}

//==============================================================================
/// standardDeviation(x:alongAxes:)
/// Returns the variance of all values are along the specified
/// axes. The result extent along the specified axes will be 1.
/// Rank is not reduced.

/// to result
/// - Parameter x: value tensor
/// - Parameter alongAxes: the axes to operate on
/// - Parameter result: the scalar tensor where the result will be written
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
@inlinable @inline(__always)
public func standardDeviation<T>(_ x: T,
                                 alongAxes axes: Vector<IndexElement>? = nil,
                                 result: inout T)
    where T: TensorView, T.Element: FloatingPoint
{
    _Streams.current.sqrt(x: variance(x, alongAxes: axes), result: &result)
}

/// to result
/// - Parameter x: value tensor
/// - Parameter alongAxes: the axes to operate on
/// - Parameter meanValue: the tensor where the mean will be written
/// - Parameter result: the scalar tensor where the result will be written
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
@inlinable @inline(__always)
public func standardDeviation<T>(_ x: T,
                                 alongAxes axes: Vector<IndexElement>? = nil,
                                 meanValue: inout T, result: inout T)
    where T: TensorView, T.Element: FloatingPoint
{
    _Streams.current.mean(x: x, along: axes, result: &meanValue)
    var varianceX = result.createDense()
    variance(x, alongAxes: axes, meanValue: &meanValue, result: &varianceX)
    _Streams.current.sqrt(x: varianceX, result: &result)
}

/// return result
/// - Parameter x: value tensor
/// - Parameter alongAxes: the axes to operate on
/// - Parameter result: the scalar tensor where the result will be written
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
@inlinable @inline(__always)
public func standardDeviation<T>(
    _ x: T, alongAxes axes: Vector<IndexElement>? = nil) -> T
    where T: TensorView, T.Element: FloatingPoint
{
    var result = x.createDense(with: [Int](repeating: 1, count: x.rank))
    Netlib.standardDeviation(x, alongAxes: axes, result: &result)
    return result
}

/// returns new view
/// - Parameter alongAxes: the axes to operate on
/// - Returns: a new tensor containing the result
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
public extension TensorView where Element: FloatingPoint {
    @inlinable @inline(__always)
    func standardDeviation(alongAxes: [Int]) -> Self {
        // turn into a vector
        let axes = Vector<IndexElement>(
            any: shape.makePositive(indices: alongAxes))
        var result = createDense()
        Netlib.standardDeviation(self, alongAxes: axes, result: &result)
        return result
    }
    
    @inlinable @inline(__always)
    func standardDeviation(alongAxes: Int...) -> Self {
        return standardDeviation(alongAxes: alongAxes)
    }
    
    @inlinable @inline(__always)
    func standardDeviation() -> Self {
        return Netlib.standardDeviation(self)
    }
    
    /// returns new view
    /// - Parameter alongAxes: the axes to operate on
    /// - Returns: a new NDTensor containing the result
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable @inline(__always)
    func standardDeviation(squeezingAxes: Int...) -> NDTensor<Element> {
        let axes = shape.makePositive(indices: squeezingAxes)
        return standardDeviation(alongAxes: squeezingAxes).squeezed(axes: axes)
    }
}

