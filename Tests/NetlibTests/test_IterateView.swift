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
        ("test_transposedMatrix", test_transposedMatrix),
        ("test_repeatingValue", test_repeatingValue),
        ("test_repeatingRow", test_repeatingRow),
        ("test_repeatingCol", test_repeatingCol),
        ("test_repeatingMatrix", test_repeatingMatrix),
        ("test_repeatingVolume", test_repeatingVolume),
        ("test_paddedVector", test_paddedVector),
        ("test_paddedMatrix", test_paddedMatrix),
    ]
    
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
//            try print(subView.formatted(numberFormat: (2,0)))
            
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
//            try print(matrix.formatted(numberFormat: (2,0)))
            
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
//            try print(matrix.formatted(numberFormat: (2,0)))
            
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

    //==========================================================================
    // test_paddedVector
    func test_paddedVector() {
        do {
            let padding = [
                Padding(before: 2, after: 3)  // col pad
            ]
            let vector = Vector<Int32>(padding: padding,
                                       padValue: -1,
                                       scalars: [Int32](0..<3))
//            try print(vector.formatted(numberFormat: (2,0)))

            let expectedValues: [Int32] = [
                -1, -1, 0, 1, 2, -1, -1, -1,
            ]

            let values = try [Int32](vector.values())
            XCTAssert(values == expectedValues, "indices do not match")

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

            let matrix = Matrix<Int32>(extents: [2,3],
                                       padding: padding,
                                       padValue: -1,
                                       scalars: [Int32](0..<6))
//            try print(matrix.formatted(numberFormat: (2,0)))

            let expectedValues: [Int32] = [
                -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1,  0,  1,  2, -1, -1, -1,
                -1, -1,  3,  4,  5, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1,
            ]

            let values = try [Int32](matrix.values())
            XCTAssert(values == expectedValues, "indices do not match")

        } catch {
            XCTFail(String(describing: error))
        }

        do {
            // edge case of 0 padding specified
            let matrix = Matrix<Int32>(extents: [2,3],
                                       padding: [Padding(0)],
                                       padValue: -1,
                                       scalars: [Int32](0..<6))
            
            let expectedValues: [Int32] = [
                0,  1,  2,
                3,  4,  5,
            ]
            
            let values = try [Int32](matrix.values())
            XCTAssert(values == expectedValues, "indices do not match")
            
        } catch {
            XCTFail(String(describing: error))
        }
    }
}
