//
//  CodableTests.swift
//  NetlibTests
//
//  Created by Edward Connell on 3/23/19.
//
import XCTest
import Foundation
@testable import Netlib

class test_TensorViewCollection: XCTestCase {
//    static var allTests = [
//        ("test_squeezed", test_squeezed),
//        ("test_transposed", test_transposed),
//        ("test_iterateRepeatedSequence", test_iterateRepeatedSequence),
//        ("test_iterateRepeatedView", test_iterateRepeatedView),
//        ("test_iteratePaddedSequence", test_iteratePaddedSequence),
//        ("test_iterateSequence", test_iterateSequence),
//        ("test_iterateShaped", test_iterateShaped),
//        ("test_perfIterateMatrixIndices", test_perfIterateMatrixIndices),
//        ("test_perfIterateMatrixValues", test_perfIterateMatrixValues),
//        ("test_perfIterateRepeatedRowMatrixIndices", test_perfIterateRepeatedRowMatrixIndices),
//        ("test_perfIterateRepeatedColMatrixIndices", test_perfIterateRepeatedColMatrixIndices),
//        ("test_perfIteratePaddedMatrixValues", test_perfIteratePaddedMatrixValues),
//    ]
//
    //==========================================================================
    // test_perfIterateMatrixValues
    func test_perfIterateVector() {
        do {
            let count: Int32 = 1024 * 1024
            let vector = Vector<Int32>(scalars: [Int32](0..<count))
//            try print(vector.formatted(numberFormat: (2,0)))

            let values = try vector.values()
            self.measure {
                for _ in values {}
            }
        } catch {
            XCTFail(String(describing: error))
        }
    }
    
//    //==========================================================================
//    // test_squeezed
//    func test_squeezed() {
//        XCTAssert(DataShape(10, 1, 4, 3, 1).squeezed().extents == [10,4,3])
//        XCTAssert(DataShape(10, 1, 4, 3, 1, 1).squeezed().extents == [10,4,3])
//        XCTAssert(DataShape(1, 1, 4, 3, 1).squeezed().extents == [4,3])
//        XCTAssert(DataShape(1, 1, 4, 3, 5).squeezed().extents == [4,3,5])
//        XCTAssert(DataShape(1, 1, 4, 1, 1, 3, 5).squeezed().extents == [4,3,5])
//
//        XCTAssert(DataShape(10, 1, 4, 3, 1).squeezed(axes: [0,4]).extents == [10,1,4,3])
//        XCTAssert(DataShape(10, 1, 4, 3, 1, 1).squeezed(axes: [0,5]).extents == [10,1,4,3,1])
//        XCTAssert(DataShape(1, 1, 4, 3, 1).squeezed(axes: [1,3]).extents == [1,4,3,1])
//        XCTAssert(DataShape(1, 1, 4, 3, 5).squeezed(axes: [3,3]).extents == [1,1,4,3,5])
//        XCTAssert(DataShape(1, 1, 4, 1, 1, 3, 5).squeezed(axes: []).extents == [1, 1, 4, 1, 1, 3, 5])
//    }
//
//    //==========================================================================
//    // test_transposed
//    func test_transposed() {
//        //        let avals = (0..<6).map { Float($0) }
//        //        let a = TensorView<Float>(extents: 2,3, scalars: avals)
//
//    }
//
//    //==========================================================================
//    // test_iterateRepeatedSequence
//    func test_iterateRepeatedSequence() {
//        do {
//            // try repeating a scalar
//            let shape = DataShape(extents: [2, 3])
//            let dataShape = DataShape(extents: [1, 1])
//            let expected = [
//                0, 0, 0,
//                0, 0, 0,
//            ]
//
//            let indices = [Int](shape.indices(repeating: dataShape))
//            XCTAssert(indices == expected, "indices do not match")
//        }
//
//        do {
//            // try repeating a row vector
//            let shape = DataShape(extents: [2, 3])
//            let dataShape = DataShape(extents: [1, 3])
//            let expected = [
//                0, 1, 2,
//                0, 1, 2,
//            ]
//
//            let indices = [Int](shape.indices(repeating: dataShape))
//            XCTAssert(indices == expected, "indices do not match")
//        }
//
//        do {
//            // try repeating a col vector
//            let shape = DataShape(extents: [3, 2])
//            let dataShape = DataShape(extents: [3, 1])
//            let expected = [
//                0, 0,
//                1, 1,
//                2, 2,
//            ]
//
//            let indices = [Int](shape.indices(repeating: dataShape))
//            XCTAssert(indices == expected, "indices do not match")
//        }
//    }
//
//    //==========================================================================
//    // test_iterateRepeatedView
//    func test_iterateRepeatedView() {
//        do {
//            // values to repeat
//            let data = Matrix<Int32>(extents: [2,2], scalars: [
//                1, 0,
//                0, 1,
//            ])
//
//            // create a virtual view and get it's values
//            let view = Matrix<Int32>(extents: [3, 4], repeating: data)
//            let values = try [Int32](view.values())
//
//            // compare
//            let expected: [Int32] = [
//                1, 0, 1, 0,
//                0, 1, 0, 1,
//                1, 0, 1, 0,
//            ]
//            XCTAssert(values == expected, "indices do not match")
//        } catch {
//            XCTFail(String(describing: error))
//        }
//
//        do {
//            // values to repeat
//            let data = Volume<Int32>(extents: [2,2,2], scalars:[
//                1, 0,
//                0, 1,
//
//                2, 3,
//                3, 2
//            ])
//
//            // create a virtual view and get it's values
//            let view = Volume<Int32>(extents: [2, 3, 4], repeating: data)
//            let values = try [Int32](view.values())
//
//            // compare
//            let expected: [Int32] = [
//                1, 0, 1, 0,
//                0, 1, 0, 1,
//                1, 0, 1, 0,
//
//                2, 3, 2, 3,
//                3, 2, 3, 2,
//                2, 3, 2, 3,
//            ]
//            XCTAssert(values == expected, "indices do not match")
//        } catch {
//            XCTFail(String(describing: error))
//        }
//    }
//
//    //==========================================================================
//    // test_iteratePaddedSequence
//    func test_iteratePaddedSequence() {
//        do {
//            // create matrix with padding
//            let padding = [
//                Padding(1), // row pad
//                Padding(before: 2, after: 3)  // col pad
//            ]
//            let m = Matrix<Int32>(extents: [2,3],
//                                  padding: padding,
//                                  padValue: -1,
//                                  scalars: [Int32](0..<6))
//
//            //            try print(m.formatted(numberFormat: (2,0)))
//
//            let indices = [Int](m.shape.indices())
//            let expectedIndices = [0, 1, 2, 3, 4, 5]
//            XCTAssert(indices == expectedIndices, "indices do not match")
//
//            let values = try [Int32](m.values())
//            let expectedValues: [Int32] = [
//                -1, -1, -1, -1, -1, -1, -1, -1,
//                -1, -1,  0,  1,  2, -1, -1, -1,
//                -1, -1,  3,  4,  5, -1, -1, -1,
//                -1, -1, -1, -1, -1, -1, -1, -1,
//            ]
//            XCTAssert(values == expectedValues, "indices do not match")
//
//        } catch {
//            XCTFail(String(describing: error))
//        }
//    }
//
//    //==========================================================================
//    // test_iterateSequence
//    func test_iterateSequence() {
//        do {
//            // try to iterate empty shape
//            let empty = Volume<Int32>()
//            for _ in empty.shape.indices() {
//                XCTFail("an empty shape should have an empty sequence")
//            }
//
//            // try volume with shape
//            let expected = [Int](0..<24)
//            let v1 = Volume<Int32>(extents: [2, 3, 4],
//                                   scalars: expected.map { Int32($0) })
//
//            let indices = [Int](v1.shape.indices())
//            XCTAssert(indices == expected, "indices do not match")
//        }
//    }
//
//    //==========================================================================
//    // test_perfIterateMatrixIndices
//    func test_perfIterateMatrixIndices() {
//        let m = Matrix<Int8>(extents: [1024, 1024])
//        self.measure {
//            for _ in m.shape.indices() {}
//        }
//    }
//
////    //==========================================================================
////    // test_perfIterateMatrixIndices
////    func test_perfSimpleSequenceBaseline() {
////        struct Simple : Sequence {
////            func makeIterator() -> SimpleIterator {
////                return SimpleIterator()
////            }
////        }
////
////        struct SimpleIterator : IteratorProtocol {
////            var pos = 0
////            let count = 1024 * 1024 * 1024
////
////            mutating func next() -> Int? {
////                let value = pos
////                pos += 1
////                return pos <= count ? value : nil
////            }
////        }
////
////        let s = Simple()
////        self.measure {
////            for value in s {
////                var a = value
////            }
////        }
////    }
//
//    //==========================================================================
//    // test_perfIterateRepeatedRowMatrixIndices
//    func test_perfIterateRepeatedRowMatrixIndices() {
//        let row = Matrix<Int8>(extents: [1, 1024])
//        let m = Matrix<Int8>(extents: [1024, 1024], repeating: row)
//        self.measure {
//            for _ in m.shape.indices() {}
//        }
//    }
//
//    //==========================================================================
//    // test_perfIterateRepeatedColMatrixIndices
//    func test_perfIterateRepeatedColMatrixIndices() {
//        let col = Matrix<Int8>(extents: [1024, 1])
//        let m = Matrix<Int8>(extents: [1024, 1024], repeating: col)
//        self.measure {
//            for _ in m.shape.indices() {}
//        }
//    }
//
//    //==========================================================================
//    // test_perfIterateMatrixValues
//    func test_perfIterateMatrixValues() {
//        let m = Matrix<Int8>(extents: [1024, 1024])
//        do {
//            let values = try m.values()
//            self.measure {
//                for _ in values {}
//            }
//        } catch {
//            XCTFail(String(describing: error))
//        }
//    }
//
//    //==========================================================================
//    // test_perfIteratePaddedMatrixValues
//    func test_perfIteratePaddedMatrixValues() {
//        // put a 1 scalar padding boundary around the Matrix
//        // as if we were going to do a 3x3 convolution
//        // by default the padValue is 0
//        let m = Matrix<Int8>(extents: [1024, 1024], padding: [Padding(1)])
//        do {
//            let values = try m.values()
//            self.measure {
//                for _ in values {}
//            }
//        } catch {
//            XCTFail(String(describing: error))
//        }
//    }
//
//    //==========================================================================
//    // test_iterateShaped
//    func test_iterateShaped() {
//        //        do {
//        //            let m = Volume<Int32>(extents: [2, 3, 4], scalars: [Int32](0..<24))
//        //            for depth in m.shape {
//        //                print("depth")
//        //                for row in depth {
//        //                    print("row")
//        //                    for index in row.tensorIndices {
//        //                        let value = try m.readOnly()[index]
//        //                        print("index: \(index) value: \(value)")
//        //                    }
//        //                }
//        //            }
//        //        } catch {
//        //            XCTFail(String(describing: error))
//        //        }
//    }
}
