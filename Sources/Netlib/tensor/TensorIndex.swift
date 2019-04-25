//******************************************************************************
//  Created by Edward Connell on 4/17/19
//  Copyright © 2019 Edward Connell. All rights reserved.
//
import Foundation

//**** This design is in flux!
//     I feel it is not fast enough


//==============================================================================
/// TensorView Collection extensions
public extension TensorView {
    var array: [Scalar] {
        return [Scalar](self.values())
    }
    
    func values() -> TensorViewCollection<Self> {
        do {
            guard _Streams.current.lastError == nil else {
                return TensorViewCollection<Self>()
            }
            return try TensorViewCollection(view: self, buffer: readOnly())
        } catch {
            _Streams.current.reportDevice(error: error)
            return TensorViewCollection<Self>()
        }
    }
    
    mutating func mutableValues() -> TensorViewMutableCollection<Self> {
        do {
            guard _Streams.current.lastError == nil else {
                return TensorViewMutableCollection<Self>()
            }
            return try TensorViewMutableCollection(view: &self,
                                                   buffer: readWrite())
        } catch {
            _Streams.current.reportDevice(error: error)
            return TensorViewMutableCollection<Self>()
        }
    }

    func deviceValues(using stream: DeviceStream) throws
        -> TensorViewCollection<Self> {
        return try TensorViewCollection(
            view: self, buffer: readOnly(using: stream))
    }
    
    mutating func mutableDeviceValues(using stream: DeviceStream) throws
        -> TensorViewMutableCollection<Self> {
        return try TensorViewMutableCollection(
            view: &self, buffer: readWrite(using: stream))
    }
    
    func value(at index: [Int]) -> Scalar {
        do {
            let buffer = try readOnly()
            let padded = shape.padded(with: padding)
            let tensorIndex = TensorIndex<Self>(self, padded, at: index)
            return tensorIndex.isPad ? padValue : buffer[tensorIndex.dataIndex]
        } catch {
            _Streams.current.reportDevice(error: error)
            return Scalar()
        }
    }

    mutating func set(value: Scalar, at index: [Int]) {
        do {
            let buffer = try readWrite()
            let padded = shape.padded(with: padding)
            let tensorIndex = TensorIndex<Self>(self, padded, at: index)
            buffer[tensorIndex.dataIndex] = value
        } catch {
            _Streams.current.reportDevice(error: error)
        }
    }
}

//==============================================================================
/// TensorViewCollection
/// returns a readonly collection view of the underlying tensorArray.
public struct TensorViewCollection<View>:
    RandomAccessCollection, DefaultInitializer where View: TensorView {
    // types
    public typealias Scalar = View.Scalar

    // properties
    private let view: View
    private let paddedViewShape: DataShape
    private let buffer: UnsafeBufferPointer<Scalar>
    public var endIndex: TensorIndex<View>
    public var count: Int { return paddedViewShape.elementCount }
    public var startIndex: TensorIndex<View> {
        return TensorIndex(view, paddedViewShape)
    }
    
    public init() {
        view = View()
        buffer = UnsafeBufferPointer<Scalar>(start: nil, count: 0)
        paddedViewShape = view.shape
        endIndex = TensorIndex(view, paddedViewShape)
    }

    public init(view: View, buffer: UnsafeBufferPointer<Scalar>) throws {
        self.view = view
        self.buffer = buffer
        paddedViewShape = view.shape.padded(with: view.padding)
        endIndex = TensorIndex(view, end: paddedViewShape.elementCount)
    }

    //--------------------------------------------------------------------------
    // Collection
    public func index(before i: TensorIndex<View>) -> TensorIndex<View> {
        fatalError()
    }
    
    public func index(after i: TensorIndex<View>) -> TensorIndex<View> {
        return i.increment()
    }

    public subscript(index: TensorIndex<View>) -> Scalar {
        return index.isPad ? view.padValue : buffer[index.dataIndex]
    }
}

//==============================================================================
/// TensorViewMutableCollection
/// returns a readonly collection view of the underlying tensorArray.
public struct TensorViewMutableCollection<View>:
    RandomAccessCollection, DefaultInitializer, MutableCollection
    where View: TensorView
{
    // types
    public typealias Scalar = View.Scalar
    
    // properties
    private var view: View
    private let paddedViewShape: DataShape
    private let buffer: UnsafeMutableBufferPointer<Scalar>
    public var endIndex: TensorIndex<View>
    public var count: Int { return paddedViewShape.elementCount }
    public var startIndex: TensorIndex<View> {
        return TensorIndex(view, paddedViewShape)
    }
    
    public init() {
        view = View()
        buffer = UnsafeMutableBufferPointer<Scalar>(start: nil, count: 0)
        paddedViewShape = view.shape
        endIndex = TensorIndex(view, paddedViewShape)
    }
    
    public init(view: inout View,
                buffer: UnsafeMutableBufferPointer<Scalar>) throws {
        self.view = view
        self.buffer = buffer
        paddedViewShape = view.shape.padded(with: view.padding)
        endIndex = TensorIndex(view, end: paddedViewShape.elementCount)
    }
    
    //--------------------------------------------------------------------------
    // Collection
    public func index(before i: TensorIndex<View>) -> TensorIndex<View> {
        fatalError()
    }
    
    public var indices: Range<TensorIndex<View>> {
        return startIndex..<endIndex
    }

    public func index(after i: TensorIndex<View>) -> TensorIndex<View> {
        return i.increment()
    }
    
    public subscript(index: TensorIndex<View>) -> Scalar {
        get {
            return index.isPad ? view.padValue : buffer[index.dataIndex]
        }
        set {
            buffer[index.dataIndex] = newValue
        }
    }
}

//==============================================================================
/// ShapePosition
public struct ShapePosition {
    /// current cummulative iterative position accross the shapes
    var current: Int
    /// the span of the extent including padding used to compute relative `end`
    let span: Int
    /// the position just after the last element
    var end: Int
}

/// This is used to track the iterated position for each dimension in the view
public struct ExtentPosition {
    /// the position for the `view` being traversed, which might be
    /// different than the data view and includes padding
    var view: ShapePosition
    /// the position for the real `data` being traveresed.
    var data: ShapePosition
    /// the current position in this dimension is padding
    var currentIsPad: Bool
    /// the parent dimension is padding
    var parentIsPad: Bool
    /// All positions before this are padding.
    /// An index of -1 is returned for each padded position
    var padBefore: Int
    /// the relative span size for the before padding
    let padBeforeSpan: Int
    /// All positions after this are padding.
    /// An index of -1 is returned for each padded position
    var padAfter: Int
    /// the relative span size for the after padding
    let padAfterSpan: Int
}

//==============================================================================
/// TensorIndex
public struct TensorIndex<T> : Strideable, Comparable where T: TensorView {
    // properties
    let tensorView: T
    let viewShape: DataShape
    let dataShape: DataShape
    var currentPosition: Int
    var position = [ExtentPosition]()
    
    // computed properties
    var viewIndex: Int { return position[lastDimension].view.current }
    var dataIndex: Int { return position[lastDimension].data.current }
    var isPad: Bool { return position[lastDimension].currentIsPad }
    var lastDimension: Int { return viewShape.lastDimension }
    var rank: Int { return viewShape.rank }

    /// start position initializer (faster)
    public init(_ tensorView: T,
                _ paddedViewShape: DataShape,
                at initialIndex: [Int]? = nil) {
        self.tensorView = tensorView
        currentPosition = 0
        viewShape = paddedViewShape
        dataShape = tensorView.dataShape
        initializePosition(
            at: initialIndex ?? [Int](repeating: 0, count: viewShape.rank))
    }

    /// end position initializer (faster)
    public init(_ tensorView: T, end: Int) {
        self.tensorView = tensorView
        currentPosition = end
        viewShape = tensorView.shape.padded(with: tensorView.padding)
        dataShape = tensorView.dataShape
    }
    
    // Equatable
    public static func == (lhs: TensorIndex, rhs: TensorIndex) -> Bool {
        assert(lhs.rank == rhs.rank)
        return lhs.currentPosition == rhs.currentPosition
    }
    
    // Comparable
    public static func < (lhs: TensorIndex, rhs: TensorIndex) -> Bool {
        assert(lhs.rank == rhs.rank)
        return lhs.currentPosition < rhs.currentPosition
    }
    
    public func increment() -> TensorIndex {
        var next = self
        next.incrementPosition(dim: lastDimension)
        return next
    }
    
    public func advanced(by n: Int) -> TensorIndex {
        var next = self
        for _ in 0..<n { next.incrementPosition(dim: lastDimension) }
        return next
    }
    
    public func distance(to other: TensorIndex<T>) -> Int {
        return other.currentPosition - currentPosition
    }
    
    //==========================================================================
    /// initializePosition(at offset:
    ///
    /// sets up the first position for indexing. This is only called
    /// once per sequence iteration.
    /// Initialization moves from outer dimension to inner (0 -> rank)
    mutating func initializePosition(at initialIndex: [Int]) {
        assert(viewShape.elementCount > 0)
        assert(initialIndex.count == viewShape.rank, "rank mismatch")
        
        // get the padding and set an increment if there is more than one
        let padding = tensorView.padding
        let padIncrement = padding.count > 1 ? 1 : 0
        var padIndex = 0
        var viewCurrent = 0
        var dataCurrent = 0

        for dim in 0..<viewShape.rank {
            // compute view position
            let stride = viewShape.strides[dim]
            viewCurrent += initialIndex[dim] * stride
            let current = viewCurrent
            let span = viewShape.extents[dim] * stride
            let end = current + span
            let beforeSpan = current + padding[padIndex].before * stride
            let afterSpan = end - padding[padIndex].after * stride
            let viewPos = ShapePosition(current: current, span: span, end: end)
            padIndex += padIncrement

            // if index 0 of any dimension is in the pad area, then all
            // contained dimensions are padding as well
            let parentIsPad = dim > 0 && position[dim - 1].currentIsPad
            let currentIsPad = parentIsPad || current < beforeSpan
            
            // find initial data starting point
            if currentIsPad {
                dataCurrent = 0
            } else {
                dataCurrent +=
                    ((initialIndex[dim] - padding[padIndex].before) %
                        dataShape.extents[dim]) * dataShape.strides[dim]
            }
            
            // setup the initial position relative to the data view
            let dataCurrent = dataCurrent
            let dataSpan = dataShape.extents[dim] * dataShape.strides[dim]
            let dataEnd = dataCurrent + dataSpan
            let dataPos = ShapePosition(current: dataCurrent,
                                        span: dataSpan, end: dataEnd)
            
            // append the fully initialized first position
            position.append(ExtentPosition(
                view: viewPos,
                data: dataPos,
                currentIsPad: currentIsPad,
                parentIsPad: parentIsPad,
                padBefore: beforeSpan,
                padBeforeSpan: beforeSpan,
                padAfter: afterSpan,
                padAfterSpan: afterSpan))
        }
        currentPosition = viewIndex
    }
    
    //==========================================================================
    /// incrementPosition(dim:
    /// increments the current position
    /// Minimal cost per: 4 cmp, 1 inc
    private mutating func incrementPosition(dim: Int) {
        //--------------------------------
        // advance the `view` position for this dimension by it's stride
        position[dim].view.current += viewShape.strides[dim]

        //--------------------------------
        // if view position is past the end
        // then advance parent dimension if there is one
        if position[dim].view.current == position[dim].view.end && dim > 0 {
            // make a recursive call to advance the parent dimension
            incrementPosition(dim: dim - 1)
            
            // rebase positions and boundaries
            let parent = position[dim - 1]
            position[dim].parentIsPad = parent.currentIsPad
            
            // update the view position
            let current = parent.view.current
            position[dim].view.current = current
            position[dim].view.end = current + position[dim].view.span

            // update the padding ranges
            position[dim].padBefore = current + position[dim].padBeforeSpan
            position[dim].padAfter = current + position[dim].padAfterSpan

            // update the data position
            let dataCurrent = parent.data.current
            position[dim].data.current = dataCurrent
            position[dim].data.end = dataCurrent + position[dim].data.span
            
        } else if !position[dim].currentIsPad {
            //--------------------------------
            // advance the `data` position for this dimension by it's stride
            // if we are not in a padded region
            // advance data index
            position[dim].data.current += dataShape.strides[dim]
            
            // if past the data end, then go back to beginning and repeat
            if position[dim].data.current == position[dim].data.end {
                position[dim].data.current -= position[dim].data.span
            }
        }
        
//        if dim == viewShape.lastDimension {
//            let dataPos = position[dim].data.current
//            print("viewPos: \(position[dim].view.current) dataPos: \(dataPos)")
//        }

        // cache the new position to simplify the Compare < function
        currentPosition = viewIndex

        // test if the current position is in a padding area
        position[dim].currentIsPad =
            position[dim].parentIsPad ||
            currentPosition < position[dim].padBefore ||
            currentPosition >= position[dim].padAfter
    }
}
