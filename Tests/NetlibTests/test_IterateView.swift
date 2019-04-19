//
//  CodableTests.swift
//  NetlibTests
//
//  Created by Edward Connell on 3/23/19.
//
import XCTest
import Foundation
@testable import Netlib

class test_IterateView: XCTestCase {
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
    // test_Vector
    func test_Vector() {
        do {
            let count: Int32 = 10
            let expected = [Int32](0..<count)
            let vector = Vector<Int32>(scalars: expected)
//            try print(vector.formatted(numberFormat: (2,0)))
            
            let values = try [Int32](vector.values())
            XCTAssert(values == expected, "values do not match")
        } catch {
            XCTFail(String(describing: error))
        }
    }

    //==========================================================================
    // test_Matrix
    func test_Matrix() {
        do {
            let expected = [Int32](0..<4)
            let matrix = Matrix<Int32>(extents: [2, 2], scalars: expected)
//            try print(matrix.formatted(numberFormat: (2,0)))
            
            let values = try [Int32](matrix.values())
            XCTAssert(values == expected, "values do not match")
        } catch {
            XCTFail(String(describing: error))
        }
        
    }
    
    //==========================================================================
    // test_Volume
    func test_Volume() {
        do {
            let expected = [Int32](0..<24)
            let volume = Volume<Int32>(extents: [2, 3, 4], scalars: expected)
//            try print(volume.formatted(numberFormat: (2,0)))
            
            let values = try [Int32](volume.values())
            XCTAssert(values == expected, "values do not match")
        } catch {
            XCTFail(String(describing: error))
        }
    }

    //==========================================================================
    // test_VectorSubView
    func test_VectorSubView() {
        do {
            let vector = Vector<Int32>(scalars: [Int32](0..<10))
            let subView = vector.view(at: [2], extents: [3])
//            try print(subView.formatted(numberFormat: (2,0)))
            
            let expected: [Int32] = [2, 3, 4]
            let values = try [Int32](subView.values())
            XCTAssert(values == expected, "values do not match")
        } catch {
            XCTFail(String(describing: error))
        }
    }
    
    //==========================================================================
    // test_MatrixSubView
    func test_MatrixSubView() {
        do {
            let matrix = Matrix<Int32>(extents: [3, 4],
                                       scalars: [Int32](0..<12))
            let subView = matrix.view(at: [1, 1], extents: [2, 2])
//            try print(subView.formatted(numberFormat: (2,0)))
            
            let expected: [Int32] = [
                5, 6,
                9, 10
            ]
            let values = try [Int32](subView.values())
            XCTAssert(values == expected, "values do not match")
        } catch {
            XCTFail(String(describing: error))
        }
    }
    
    //==========================================================================
    // test_VolumeSubView
    func test_VolumeSubView() {
        do {
            let volume = Volume<Int32>(extents: [3, 3, 4],
                                       scalars: [Int32](0..<36))
            let subView = volume.view(at: [1, 1, 1], extents: [2, 2, 3])
            try print(subView.formatted(numberFormat: (2,0)))
            
            let expected: [Int32] = [
                17, 18, 19,
                21, 22, 23,
                
                29, 30, 31,
                33, 34, 35,
            ]
            let values = try [Int32](subView.values())
            XCTAssert(values == expected, "values do not match")
        } catch {
            XCTFail(String(describing: error))
        }
    }
    
    //==========================================================================
    // test_perfVector
    func test_perfVector() {
        do {
            let count: Int32 = 512 * 512
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
    
    //==========================================================================
    // test_transposedMatrix
    func test_transposedMatrix() {
        //        let avals = (0..<6).map { Float($0) }
        //        let a = TensorView<Float>(extents: 2,3, scalars: avals)

    }

    //==========================================================================
    // test_repeatingValue
    func test_repeatingValue() {
        do {
            // try repeating a scalar
            let value = Matrix<Int32>(extents: [1, 1], scalars: [42])
            let matrix = Matrix<Int32>(extents: [2, 3], repeating: value)
//            try print(vector.formatted(numberFormat: (2,0)))

            let expected: [Int32] = [
                42, 42, 42,
                42, 42, 42,
            ]

            let values = try [Int32](matrix.values())
            XCTAssert(values == expected, "values do not match")
        } catch {
            XCTFail(String(describing: error))
        }
    }
    
    //==========================================================================
    // test_repeatingRow
    func test_repeatingRow() {
        do {
            // try repeating a row vector
            let row = Matrix<Int32>(extents: [1, 3], scalars: [Int32](0..<3))
            let matrix = Matrix<Int32>(extents: [2, 3], repeating: row)
            try print(matrix.formatted(numberFormat: (2,0)))
            
            let expected: [Int32] = [
                0, 1, 2,
                0, 1, 2,
            ]
            
            let values = try [Int32](matrix.values())
            XCTAssert(values == expected, "values do not match")
        } catch {
            XCTFail(String(describing: error))
        }
    }
    
    //==========================================================================
    // test_repeatingCol
    func test_repeatingCol() {
        do {
            // try repeating a row vector
            let col = Matrix<Int32>(extents: [3, 1], scalars: [Int32](0..<3))
            let matrix = Matrix<Int32>(extents: [3, 2], repeating: col)
            try print(matrix.formatted(numberFormat: (2,0)))
            
            let expected: [Int32] = [
                0, 0,
                1, 1,
                2, 2,
            ]
            
            let values = try [Int32](matrix.values())
            XCTAssert(values == expected, "values do not match")
        } catch {
            XCTFail(String(describing: error))
        }
    }
    

    //==========================================================================
    // test_repeatingMatrix
    func test_repeatingMatrix() {
        do {
            let pattern = Matrix<Int32>(extents: [2,2], scalars: [
                1, 0,
                0, 1,
            ])
            
            let matrix = Matrix<Int32>(extents: [3, 4], repeating: pattern)
            let expected: [Int32] = [
                1, 0, 1, 0,
                0, 1, 0, 1,
                1, 0, 1, 0,
            ]
            
            let values = try [Int32](matrix.values())
            XCTAssert(values == expected, "values do not match")
        } catch {
            XCTFail(String(describing: error))
        }
    }
    
    //==========================================================================
    // test_repeatingVolume
    func test_repeatingVolume() {
        do {
            let pattern = Volume<Int32>(extents: [2,2,2], scalars:[
                1, 0,
                0, 1,
                
                2, 3,
                3, 2
            ])
            
            // create a virtual view and get it's values
            let volume = Volume<Int32>(extents: [2, 3, 4], repeating: pattern)
            let expected: [Int32] = [
                1, 0, 1, 0,
                0, 1, 0, 1,
                1, 0, 1, 0,
                
                2, 3, 2, 3,
                3, 2, 3, 2,
                2, 3, 2, 3,
            ]

            let values = try [Int32](volume.values())
            XCTAssert(values == expected, "indices do not match")
        } catch {
            XCTFail(String(describing: error))
        }
    }

//    //==========================================================================
//    // test_paddedVector
//    func test_paddedVector() {
//        do {
//            let padding = [
//                Padding(before: 2, after: 3)  // col pad
//            ]
//            let vector = Vector<Int32>(padding: padding,
//                                       padValue: -1,
//                                       scalars: [Int32](0..<3))
//            try print(vector.formatted(numberFormat: (2,0)))
//            
//            let expectedValues: [Int32] = [
//                -1, -1,  0,  1,  2, -1, -1, -1,
//            ]
//
//            let values = try [Int32](vector.values())
//            XCTAssert(values == expectedValues, "indices do not match")
//            
//        } catch {
//            XCTFail(String(describing: error))
//        }
//    }
//
//    //==========================================================================
//    // test_paddedMatrix
//    func test_paddedMatrix() {
//        do {
//            // create matrix with padding
//            let padding = [
//                Padding(1),                   // row pad
//                Padding(before: 2, after: 3)  // col pad
//            ]
//
//            let matrix = Matrix<Int32>(extents: [2,3],
//                                       padding: padding,
//                                       padValue: -1,
//                                       scalars: [Int32](0..<6))
//            try print(matrix.formatted(numberFormat: (2,0)))
//
//            let expectedValues: [Int32] = [
//                -1, -1, -1, -1, -1, -1, -1, -1,
//                -1, -1,  0,  1,  2, -1, -1, -1,
//                -1, -1,  3,  4,  5, -1, -1, -1,
//                -1, -1, -1, -1, -1, -1, -1, -1,
//            ]
//
//            let values = try [Int32](matrix.values())
//            XCTAssert(values == expectedValues, "indices do not match")
//
//        } catch {
//            XCTFail(String(describing: error))
//        }
//    }

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
