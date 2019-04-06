//
//  CodableTests.swift
//  NetlibTests
//
//  Created by Edward Connell on 3/23/19.
//
import XCTest
import Foundation
@testable import Netlib

class test_DataShape: XCTestCase {
    static var allTests = [
        ("test_squeezed", test_squeezed),
        ("test_transposed", test_transposed),
        ("test_iterateSequence", test_iterateSequence),
        ("test_iterateShaped", test_iterateShaped),
    ]

    //==========================================================================
    // test_squeezed
    func test_squeezed() {
        XCTAssert(DataShape(10, 1, 4, 3, 1).squeezed().extents == [10,4,3])
        XCTAssert(DataShape(10, 1, 4, 3, 1, 1).squeezed().extents == [10,4,3])
        XCTAssert(DataShape(1, 1, 4, 3, 1).squeezed().extents == [4,3])
        XCTAssert(DataShape(1, 1, 4, 3, 5).squeezed().extents == [4,3,5])
        XCTAssert(DataShape(1, 1, 4, 1, 1, 3, 5).squeezed().extents == [4,3,5])
        
        XCTAssert(DataShape(10, 1, 4, 3, 1).squeezed(axes: [0,4]).extents == [10,1,4,3])
        XCTAssert(DataShape(10, 1, 4, 3, 1, 1).squeezed(axes: [0,5]).extents == [10,1,4,3,1])
        XCTAssert(DataShape(1, 1, 4, 3, 1).squeezed(axes: [1,3]).extents == [1,4,3,1])
        XCTAssert(DataShape(1, 1, 4, 3, 5).squeezed(axes: [3,3]).extents == [1,1,4,3,5])
        XCTAssert(DataShape(1, 1, 4, 1, 1, 3, 5).squeezed(axes: []).extents == [1, 1, 4, 1, 1, 3, 5])
    }
    
    //==========================================================================
    // test_transposed
    func test_transposed() {
//        let avals = (0..<6).map { Float($0) }
//        let a = TensorView<Float>(extents: 2,3, scalars: avals)
        
    }

    //==========================================================================
    // test_iterateSequence
    func test_iterateSequence() {
        // try to iterate empty shape
        let empty = VolumeTensor<Int32>()
        for _ in empty.shape.relativeIndices {
            XCTFail("an empty shape should have an empty sequence")
        }
        
        // try volume with shape
//        let expected = [Int](0..<24)
        let expected = [Int](0..<8)
        let m = VolumeTensor<Int32>(extents: [1, 2, 4],
                                    scalars: expected.map { Int32($0) })
        let indices = [Int](m.shape.relativeIndices)
        XCTAssert(indices == expected, "indices do not match")
    }

    //==========================================================================
    // test_iterateShaped
    func test_iterateShaped() {
//        do {
//            let m = VolumeTensor<Int32>(extents: [2, 3, 4], scalars: [Int32](0..<24))
//            for depth in m.shape {
//                print("depth")
//                for row in depth {
//                    print("row")
//                    for index in row.tensorIndices {
//                        let value = try m.readOnly()[index]
//                        print("index: \(index) value: \(value)")
//                    }
//                }
//            }
//        } catch {
//            XCTFail(String(describing: error))
//        }
    }
}
