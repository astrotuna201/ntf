//******************************************************************************
//  Created by Edward Connell on 4/17/19
//  Copyright © 2019 Edward Connell. All rights reserved.
//
import Foundation

//==============================================================================
/// TensorView Collection extensions
public extension TensorView {
    func values() throws -> TensorViewCollection<Scalar> {
        return try TensorViewCollection<Scalar>(buffer: readOnly(),
                                                padValue: padValue ?? Scalar())
    }

//    func mutableValues() throws -> TensorViewCollection<Scalar> {
//        return try TensorViewCollection<Scalar>(buffer: readWrite())
//    }
}

//==============================================================================
/// TensorViewCollection
/// returns a readonly collection view of the underlying tensorData.
public struct TensorViewCollection<Scalar>: Collection {
    // properties
    let buffer: UnsafeBufferPointer<Scalar>
    let padValue: Scalar
    
    public init(buffer: UnsafeBufferPointer<Scalar>, padValue: Scalar) {
        self.buffer = buffer
        self.padValue = padValue
    }

    //--------------------------------------------------------------------------
    // Collection
    public var startIndex: TensorIndex { return TensorIndex() }
    public var endIndex: TensorIndex { return TensorIndex() }

    public func index(after i: TensorIndex) -> TensorIndex {
        return i.next()
    }

    public subscript(index: TensorIndex) -> Scalar {
        return index.data < 0 ? padValue : buffer[index.data]
    }
}


public struct TensorIndex : Comparable {
    public var shape: Int
    public var data: Int
    
    public init() {
        shape = 0
        data = 0
    }
    
    public static func < (lhs: TensorIndex, rhs: TensorIndex) -> Bool {
        return lhs.shape < rhs.shape
    }
    
    public func next() -> TensorIndex {
        fatalError()
    }
}

