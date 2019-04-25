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
        let count: Int32 = 10
        let expected = [Int32](0..<count)
        let vector = Vector<Int32>(scalars: expected)
//        print(vector.formatted((2,0)))
        
        let values = [Int32](vector.values())
        XCTAssert(values == expected, "values do not match")
    }

    //==========================================================================
    // test_Matrix
    func test_Matrix() {
            let expected = [Int32](0..<4)
            let matrix = Matrix<Int32>((2, 2), scalars: expected)
//            print(matrix.formatted((2,0)))
        
            let values = [Int32](matrix.values())
            XCTAssert(values == expected, "values do not match")
    }
    
    //==========================================================================
    // test_Volume
    func test_Volume() {
        let expected = [Int32](0..<24)
        let volume = Volume<Int32>((2, 3, 4), scalars: expected)
        //            print(volume.formatted((2,0)))
        
        let values = [Int32](volume.values())
        XCTAssert(values == expected, "values do not match")
    }

    //==========================================================================
    // test_VectorSubView
    func test_VectorSubView() {
        let vector = Vector<Int32>(scalars: [Int32](0..<10))
        let subView = vector.view(at: [2], extents: [3])
        //            print(subView.formatted((2,0)))
        
        let expected: [Int32] = [2, 3, 4]
        let values = [Int32](subView.values())
        XCTAssert(values == expected, "values do not match")
    }
    
    //==========================================================================
    // test_MatrixSubView
    func test_MatrixSubView() {
        let matrix = Matrix<Int32>((3, 4), sequence: 0..<12)
        let subView = matrix.view(at: [1, 1], extents: [2, 2])
//        print(subView.formatted((2,0)))
        
        let expected: [Int32] = [
            5, 6,
            9, 10
        ]
        let values = [Int32](subView.values())
        XCTAssert(values == expected, "values do not match")
    }
    
    //==========================================================================
    // test_VolumeSubView
    func test_VolumeSubView() {
        let volume = Volume<Int32>((3, 3, 4), sequence: 0..<36)
        let subView = volume.view(at: [1, 1, 1], extents: [2, 2, 3])
        //            print(subView.formatted((2,0)))
        
        let expected: [Int32] = [
            17, 18, 19,
            21, 22, 23,
            
            29, 30, 31,
            33, 34, 35,
        ]
        let values = [Int32](subView.values())
        XCTAssert(values == expected, "values do not match")
    }
    
    //==========================================================================
    // test_perfVector
    func test_perfVector() {
        #if RELEASE
        let count = 512 * 512
        let vector = Vector<Int32>(sequence: 0..<count)
        //            print(vector.formatted((2,0)))
        
        let values = vector.values()
        self.measure {
            for _ in values {}
        }
        #endif
    }

    //==========================================================================
    // test_repeatingValue
    func test_repeatingValue() {
        // try repeating a scalar
        let value = Matrix<Int32>((1, 1), scalars: [42])
        let matrix = Matrix<Int32>((2, 3), repeating: value)
        //            print(vector.formatted((2,0)))
        
        let expected: [Int32] = [
            42, 42, 42,
            42, 42, 42,
        ]
        
        let values = [Int32](matrix.values())
        XCTAssert(values == expected, "values do not match")
    }
    
    //==========================================================================
    // test_repeatingRow
    func test_repeatingRow() {
        // try repeating a row vector
        let row = Matrix<Int32>((1, 3), sequence: 0..<3)
        let matrix = Matrix<Int32>((2, 3), repeating: row)
        //            print(matrix.formatted((2,0)))
        
        let expected: [Int32] = [
            0, 1, 2,
            0, 1, 2,
        ]
        
        let values = [Int32](matrix.values())
        XCTAssert(values == expected, "values do not match")
    }
    
    //==========================================================================
    // test_repeatingCol
    func test_repeatingCol() {
        // try repeating a row vector
        let col = Matrix<Int32>((3, 1), sequence: 0..<3)
        let matrix = Matrix<Int32>((3, 2), repeating: col)
        //            print(matrix.formatted((2,0)))
        
        let expected: [Int32] = [
            0, 0,
            1, 1,
            2, 2,
        ]
        
        XCTAssert(matrix.array == expected, "values do not match")
    }
    

    //==========================================================================
    // test_repeatingMatrix
    func test_repeatingMatrix() {
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
        
        XCTAssert(matrix.array == expected, "values do not match")
    }
    
    //==========================================================================
    // test_repeatingVolume
    func test_repeatingVolume() {
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
        
        XCTAssert(volume.array == expected, "values do not match")
    }

    //==========================================================================
    // test_paddedVector
    func test_paddedVector() {
        // col pad
        let padding = [Padding(before: 2, after: 3)]

        let vector = Vector<Int32>(padding: padding, padValue: -1,
                                   sequence: 0..<3)
        //            print(vector.formatted((2,0)))
        
        let expected: [Int32] = [-1, -1, 0, 1, 2, -1, -1, -1]
        XCTAssert(vector.array == expected, "values do not match")
    }

    //==========================================================================
    // test_paddedMatrix
    func test_paddedMatrix() {
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
        
        XCTAssert(matrix.array == expected, "values do not match")

        // edge case of 0 padding specified
        let matrix2 = Matrix<Int32>((2, 3),
                                    padding: [Padding(0)],
                                    padValue: -1,
                                    sequence: 0..<6)

        let expected2: [Int32] = [
            0,  1,  2,
            3,  4,  5,
        ]
        
        XCTAssert(matrix2.array == expected2, "values do not match")
    }
}
