//******************************************************************************
//  Created by Edward Connell on 4/7/19
//  Copyright © 2019 Edward Connell. All rights reserved.
//
import Foundation

//==============================================================================
// TensorViewSequence
public struct TensorViewSequence<T>: Sequence where T: TensorView {
    // properties
    let view: T
    let tensorDataBuffer: UnsafeBufferPointer<T.Scalar>

    // initializers
    public init(view: T) throws {
        self.view = view
        try tensorDataBuffer = view.readOnly()
    }
    
    public func makeIterator() -> TensorViewSequenceIterator<T> {
        return TensorViewSequenceIterator(view: view, buffer: tensorDataBuffer)
    }
}

//==============================================================================
/// DataShapeSequenceIterator
/// This iterates the tensorData indexes described by
/// an N dimensional DataShape as a single linear Sequence
public struct TensorViewSequenceIterator<T>: IteratorProtocol
where T: TensorView {
    // properties
    let padValue: T.Scalar
    var indexIterator: DataShapeSequenceIterator
    let tensorDataBuffer: UnsafeBufferPointer<T.Scalar>

    // initializers
    init(view: T, buffer: UnsafeBufferPointer<T.Scalar>) {
        padValue = view.padValue!
        tensorDataBuffer = buffer
        indexIterator = DataShapeSequenceIterator(view: view.shape,
                                                  at: view.viewOffset,
                                                  repeating: view.dataShape,
                                                  with: view.padding)
    }

    /// next
    public mutating func next() -> T.Scalar? {
        guard let index = indexIterator.next() else { return nil }
        return index < 0 ? padValue : tensorDataBuffer[index]
    }
}

//==============================================================================
// default TensorView Sequence implementation
public extension TensorView {
    //--------------------------------------------------------------------------
    /// values
    /// Returns a sequential iterator of all values
    func values() throws -> TensorViewSequence<Self> {
        return try TensorViewSequence(view: self)
    }
}
