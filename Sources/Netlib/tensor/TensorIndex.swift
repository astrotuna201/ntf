////******************************************************************************
////  Created by Edward Connell on 5/4/19
////  Copyright © 2019 Edward Connell. All rights reserved.
////
//import Foundation
//
////==============================================================================
///// TensorIndexing
//public protocol TensorIndexing where Self: TensorView {
//    /// returns a collection of values in spatial order
//    func values() throws -> TensorValueCollection<Self>
////    /// returns a collection of mutable values in spatial order
////    mutating func mutableValues() throws -> ViewMutableCollection
//}
//
////==============================================================================
///// _TensorValueCollection
//public protocol _TensorValueCollection: RandomAccessCollection where
//    Index: TensorIndex & Strideable & Comparable
//{
//    associatedtype View: TensorView
//    associatedtype Scalar = View.Scalar
//
//    var buffer: UnsafeBufferPointer<Scalar> { get }
//    var paddedShape: DataShape { get }
//}
//
//public extension _TensorValueCollection {
//    //--------------------------------------------------------------------------
//    // Collection
//    func index(before i: Index) -> Index {
//        fatalError()
//    }
//
//    func index(after i: Index) -> Index {
//        fatalError()
//    }
//
//    subscript(index: Index) -> Scalar {
//        fatalError()
//    }
//
//    func isPad(_ index: Index) -> Bool {
//        return false
//    }
//}
//
//public extension _TensorValueCollection where
//View: MatrixView, Index == MatrixIndex
//{
//    //--------------------------------------------------------------------------
//    // Collection
//    func index(before i: Index) -> Index {
//        fatalError()
//    }
//
//    func index(after i: Index) -> Index {
//        fatalError()
//    }
//
//    subscript(index: Index) -> Scalar {
//        fatalError()
//    }
//}
//
//==========================================================================
public protocol TensorIndex: Strideable {
    // types
    associatedtype Scalar
    typealias AdvanceFn = (Self, _ by: Int) -> Self

    // properties
    var advanceFn: AdvanceFn { get }
    var dataIndex: Int { get }
    var isPad: Bool { get }
}

public extension TensorIndex {
    // Equatable
    static func == (lhs: Self, rhs: Self) -> Bool {
        return lhs.dataIndex == rhs.dataIndex
    }

    // Comparable
    static func < (lhs: Self, rhs: Self) -> Bool {
        return lhs.dataIndex < rhs.dataIndex
    }

    // Strideable
    func advanced(by n: Int) -> Self {
        return advanceFn(self, n)
    }

    func distance(to other: Self) -> Int {
        return other.dataIndex - dataIndex
    }
}

////==============================================================================
///// TensorValueCollection
//public struct TensorValueCollection<View>:
//    _TensorValueCollection
//    where View: TensorView
//{
//    // properties
//    public let view: View
//    public let paddedShape: DataShape
//    public let buffer: UnsafeBufferPointer<View.Scalar>
//    public var endIndex: Index
//    public var count: Int { return paddedShape.elementCount }
//    public var startIndex: Index {
//        return Index(fn: Index.advance(index:))
//    }
//
//    public init(view: View, buffer: UnsafeBufferPointer<Scalar>) throws {
//        self.view = view
//        self.buffer = buffer
//        paddedShape = view.shape.padded(with: view.padding)
//        endIndex = Index(view, end: paddedShape.elementCount)
//    }
//}
