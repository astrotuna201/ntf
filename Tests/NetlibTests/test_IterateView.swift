//******************************************************************************
//  Created by Edward Connell on 3/23/19
//  Copyright © 2019 Connell Research. All rights reserved.
//
import XCTest
import Foundation
@testable import Netlib

class test_IterateView: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_Vector", test_Vector),
        ("test_Matrix", test_Matrix),
        ("test_Volume", test_Volume),
        ("test_VectorSubView", test_VectorSubView),
        ("test_MatrixSubView", test_MatrixSubView),
        ("test_VolumeSubView", test_VolumeSubView),
        ("test_perfVector", test_perfVector),
        ("test_perfMatrix", test_perfMatrix),
        ("test_perfVolume", test_perfVolume),
        ("test_perfIndexCopy", test_perfIndexCopy),
        ("test_repeatingValue", test_repeatingValue),
        ("test_repeatingRow", test_repeatingRow),
        ("test_repeatingCol", test_repeatingCol),
        ("test_repeatingMatrix", test_repeatingMatrix),
        ("test_repeatingVolume", test_repeatingVolume),
        ("test_paddedVector", test_paddedVector),
        ("test_paddedMatrix", test_paddedMatrix),
        ("test_paddedRepeatedMatrix", test_paddedRepeatedMatrix),
    ]
    
    //==========================================================================
    // test_Vector
    func test_Vector() {
        do {
            let count: Int32 = 10
            let expected = [Int32](0..<count)
            let vector = Vector<Int32>(scalars: expected)
            //        print(vector.formatted((2,0)))
            
            let values = try vector.array()
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
            let matrix = Matrix<Int32>((2, 2), scalars: expected)
//                        print(matrix.formatted((2,0)))
            
            let values = try matrix.array()
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
            let volume = Volume<Int32>((2, 3, 4), scalars: expected)
            //            print(volume.formatted((2,0)))
            
            let values = try volume.array()
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
            let view = vector.view(at: [2], extents: [3])
            //            print(subView.formatted((2,0)))
            
            let expected: [Int32] = [2, 3, 4]
            let values = try view.array()
            XCTAssert(values == expected, "values do not match")
        } catch {
            XCTFail(String(describing: error))
        }
    }
    
    //==========================================================================
    // test_MatrixSubView
    func test_MatrixSubView() {
        do {
            let matrix = Matrix<Int32>((3, 4), sequence: 0..<12)
            let view = matrix.view(at: [1, 1], extents: [2, 2])
                    print(view.formatted((2,0)))
            
            let expected: [Int32] = [
                5, 6,
                9, 10
            ]
            let values = try view.array()
            XCTAssert(values == expected, "values do not match")
        } catch {
            XCTFail(String(describing: error))
        }
    }
    
    //==========================================================================
    // test_VolumeSubView
    func test_VolumeSubView() {
        do {
            let volume = Volume<Int32>((3, 3, 4), sequence: 0..<36)
            let view = volume.view(at: [1, 1, 1], extents: [2, 2, 3])
            //            print(subView.formatted((2,0)))
            
            let expected: [Int32] = [
                17, 18, 19,
                21, 22, 23,
                
                29, 30, 31,
                33, 34, 35,
            ]
            let values = try view.array()
            XCTAssert(values == expected, "values do not match")
        } catch {
            XCTFail(String(describing: error))
        }
    }
    
    //==========================================================================
    // test_perfVector
    func test_perfVector() {
        #if !DEBUG
        do {
            let count = 1024 * 1024
            let vector = Vector<Int32>(count: count, sequence: 0..<count)
            //            print(matrix.formatted((2,0)))
            let values = try vector.values()
            
            self.measure {
                for _ in values {}
            }
        } catch {
            XCTFail(String(describing: error))
        }
        #endif
    }
    
    //==========================================================================
    // test_perfMatrix
    func test_perfMatrix() {
        #if !DEBUG
        do {
            let rows = 1024
            let cols = 1024

            let matrix = Matrix<Int32>((rows, cols), sequence: 0..<(rows * cols))
            //            print(matrix.formatted((2,0)))
            
            let values = try matrix.values()

            self.measure {
                for _ in values {}
            }
        } catch {
            XCTFail(String(describing: error))
        }
        #endif
    }
    
    //==========================================================================
    // test_perfVolume
    func test_perfVolume() {
        #if !DEBUG
        do {
            let depths = 4
            let rows = 512
            let cols = 512
            
            let matrix = Volume<Int32>((depths, rows, cols),
                                       sequence: 0..<(depths * rows * cols))
            //            print(matrix.formatted((2,0)))
            
            let values = try matrix.values()
            
            self.measure {
                for _ in values {}
            }
        } catch {
            XCTFail(String(describing: error))
        }
        #endif
    }
    
    //==========================================================================
    // test_perfIndexCopy
    func test_perfIndexCopy() {
        #if !DEBUG
        var m = Matrix<Int32>((1024, 1024)).startIndex
        
        self.measure {
            for _ in 0..<1000000 {
                m = m.increment()
//                m = m.advanced(by: 1)
            }
        }
        #endif
    }
    //==========================================================================
    // test_repeatingValue
    func test_repeatingValue() {
        do {
            // try repeating a scalar
            let value = Matrix<Int32>((1, 1), scalars: [42])
            let matrix = Matrix<Int32>((2, 3), repeating: value)
            //            print(vector.formatted((2,0)))
            
            let expected: [Int32] = [
                42, 42, 42,
                42, 42, 42,
            ]
            
            let values = try matrix.array()
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
            let row = Matrix<Int32>((1, 3), sequence: 0..<3)
            let matrix = Matrix<Int32>((2, 3), repeating: row)
            //            print(matrix.formatted((2,0)))
            
            let expected: [Int32] = [
                0, 1, 2,
                0, 1, 2,
            ]
            
            let values = try matrix.array()
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
            let col = Matrix<Int32>((3, 1), sequence: 0..<3)
            let matrix = Matrix<Int32>((3, 2), repeating: col)
                        print(matrix.formatted((2,0)))
            
            let expected: [Int32] = [
                0, 0,
                1, 1,
                2, 2,
            ]
            
            let values = try matrix.array()
            XCTAssert(values == expected, "values do not match")
        } catch {
            XCTFail(String(describing: error))
        }
    }
    
    //==========================================================================
    // test_repeatingMatrix
    func test_repeatingMatrix() {
        do {
            let pattern = Matrix<Int32>((2, 2), scalars: [
                1, 0,
                0, 1,
                ])
            
            let matrix = Matrix<Int32>((3, 4), repeating: pattern)
            let expected: [Int32] = [
                1, 0, 1, 0,
                0, 1, 0, 1,
                1, 0, 1, 0,
            ]

            let values = try matrix.array()
            XCTAssert(values == expected, "values do not match")
        } catch {
            XCTFail(String(describing: error))
        }
    }
    
    //==========================================================================
    // test_repeatingVolume
    func test_repeatingVolume() {
        do {
            let pattern = Volume<Int32>((2,2,2), scalars:[
                1, 0,
                0, 1,
                
                2, 3,
                3, 2
                ])
            
            // create a virtual view and get it's values
            let volume = Volume<Int32>((2, 3, 4), repeating: pattern)
            let expected: [Int32] = [
                1, 0, 1, 0,
                0, 1, 0, 1,
                1, 0, 1, 0,
                
                2, 3, 2, 3,
                3, 2, 3, 2,
                2, 3, 2, 3,
            ]
            
            let values = try volume.array()
            XCTAssert(values == expected, "values do not match")
        } catch {
            XCTFail(String(describing: error))
        }
    }
    
    //==========================================================================
    // test_paddedVector
    func test_paddedVector() {
        do {
            // col pad
            let padding = [Padding(before: 2, after: 3)]
            
            let vector = Vector<Int32>(padding: padding, padValue: -1,
                                       sequence: 0..<3)
            //            print(vector.formatted((2,0)))
            
            let expected: [Int32] = [-1, -1, 0, 1, 2, -1, -1, -1]
            let values = try vector.array()
            XCTAssert(values == expected, "values do not match")
        } catch {
            XCTFail(String(describing: error))
        }
    }
    
    //==========================================================================
    // test_paddedMatrix
    func test_paddedMatrix() {
        do {
            // create matrix with padding
            let padding = [
                Padding(1),                   // row pad
                Padding(before: 2, after: 3)  // col pad
            ]
            
            let matrix = Matrix<Int32>((2, 3),
                                       padding: padding,
                                       padValue: -1,
                                       sequence: 0..<6)
            print(matrix.formatted((2,0)))
            
            let expected: [Int32] = [
                -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1,  0,  1,  2, -1, -1, -1,
                -1, -1,  3,  4,  5, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1,
            ]
            
            let values1 = try matrix.array()
            XCTAssert(values1 == expected, "values do not match")

            // edge case of 0 padding specified
            let matrix2 = Matrix<Int32>((2, 3),
                                        padding: [Padding(0)],
                                        padValue: -1,
                                        sequence: 0..<6)
            
            let expected2: [Int32] = [
                0,  1,  2,
                3,  4,  5,
            ]
            
            let values2 = try matrix2.array()
            XCTAssert(values2 == expected2, "values do not match")
        } catch {
            XCTFail(String(describing: error))
        }
    }
    
    //==========================================================================
    // test_paddedRepeatedMatrix
    func test_paddedRepeatedMatrix() {
        // TODO: implement negative coordinates to create padding
//        do {
//            let column = Matrix<Int32>((3, 1), sequence: 0..<3)
//            let matrix = Matrix<Int32>((3, 3), repeating: column)
//            let padded = matrix.view(at: [-1, -2], extents: [5, 8])
////            print(matrix.formatted((2,0)))
//
//            let expected: [Int32] = [
//                -1, -1, -1, -1, -1, -1, -1, -1,
//                -1, -1,  0,  0,  0, -1, -1, -1,
//                -1, -1,  1,  1,  1, -1, -1, -1,
//                -1, -1,  2,  2,  2, -1, -1, -1,
//                -1, -1, -1, -1, -1, -1, -1, -1,
//            ]
//            let values = try padded.array()
//            XCTAssert(values == expected, "values do not match")
//
//        } catch {
//            XCTFail(String(describing: error))
//        }
    }
}
