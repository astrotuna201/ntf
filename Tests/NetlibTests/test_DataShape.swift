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
        ("test_iterateModuloSequence", test_iterateModuloSequence),
        ("test_iterateModuloView", test_iterateModuloView),
        ("test_iteratePaddedSequence", test_iteratePaddedSequence),
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
    // test_iterateModuloSequence
    func test_iterateModuloSequence() {
        do {
            // try broadcasting a scalar
            let shape = DataShape(extents: [2, 3])
            let dataShape = DataShape(extents: [1, 1])
            let expected = [
                0, 0, 0,
                0, 0, 0,
            ]

            let indices = [Int](shape.indices(modulo: dataShape))
            XCTAssert(indices == expected, "indices do not match")
        }
        
        do {
            // try broadcasting a row vector
            let shape = DataShape(extents: [2, 3])
            let dataShape = DataShape(extents: [1, 3])
            let expected = [
                0, 1, 2,
                0, 1, 2,
            ]
            
            let indices = [Int](shape.indices(modulo: dataShape))
            XCTAssert(indices == expected, "indices do not match")
        }
        
        do {
            // try broadcasting a col vector
            let shape = DataShape(extents: [3, 2])
            let dataShape = DataShape(extents: [3, 1])
            let expected = [
                0, 0,
                1, 1,
                2, 2,
            ]
            
            let indices = [Int](shape.indices(modulo: dataShape))
            XCTAssert(indices == expected, "indices do not match")
        }
    }
    
    //==========================================================================
    // test_iterateModuloView
    func test_iterateModuloView() {
        do {
            // try broadcasting a pattern of values
            let data: [Int32] = [
                1, 0,
                0, 1,
            ]
            let pattern = MatrixTensor<Int32>(extents: [2,2], scalars: data)
            let _ = MatrixTensor<Int32>(extents: [3, 4], modulo: pattern)

//            let expected = [
//                0, 0, 0,
//                0, 0, 0,
//            ]
            
//            let values = [Int](view.indices(modulo: dataShape))
//            XCTAssert(indices == expected, "indices do not match")
        }
    }
    
    //==========================================================================
    // test_iteratePaddedSequence
    func test_iteratePaddedSequence() {
        // try volume with shape and padding
        let v2 = VolumeTensor<Int32>(extents: [1, 3, 4],
                                     padding: [Padding(before: 2, after: 3)],
                                     scalars: [Int32](0..<12))
        let expectedPadded = [
            -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1,  0,  1,  2,  3, -1, -1, -1,
            -1, -1,  4,  5,  6,  7, -1, -1, -1,
            -1, -1,  8,  9, 10, 11, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1,
        ]
        
        let v2Indices = [Int](v2.shape.indices())
        XCTAssert(v2Indices == expectedPadded, "indices do not match")
    }
    
    //==========================================================================
    // test_iterateSequence
    func test_iterateSequence() {
        // try to iterate empty shape
        let empty = VolumeTensor<Int32>()
        for _ in empty.shape.indices() {
            XCTFail("an empty shape should have an empty sequence")
        }

        // try volume with shape
        let expected = [Int](0..<24)
        let v1 = VolumeTensor<Int32>(extents: [2, 3, 4],
                                     scalars: expected.map { Int32($0) })
        let indices = [Int](v1.shape.indices())
        XCTAssert(indices == expected, "indices do not match")
    }

    func testPerformanceExample() {
        let scalars: [Float] = (0..<24).map { Float($0) }
        // This is an example of a performance test case.
        self.measure {
            for _ in 0..<10000 {
                let _ = VolumeTensor<Float>(extents: [2, 3, 4], scalars: scalars)
            }
        }
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
