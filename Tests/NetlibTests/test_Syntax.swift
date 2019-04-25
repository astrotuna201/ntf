//******************************************************************************
//  Created by Edward Connell on 4/24/19
//  Copyright © 2019 Connell Research. All rights reserved.
//
import XCTest
import Foundation

@testable import Netlib

class test_Syntax: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_simple", test_simple),
    ]
    
    //==========================================================================
    // test_simple
    func test_simple() {
        //--------------------------------
        // initialize a matrix with a sequence and take the sum
        // use shortcut syntax for extents
        do {
            let matrix = Matrix<Float>((3, 5), sequence: 0..<15)
            print(matrix.formatted((2,0)))
            let sum = matrix.sum().scalarValue()
            XCTAssert(sum == 105.0)
        }
        
        //--------------------------------
        // Select and sum a 3D sub region
        // - initialize a volume using explicit extents
        // - fill with indexes on the default device
        // - create a sub view and take the sum on the device
        // - return the scalar value back to the app thread
        do {
            let volume = Volume<Int32>((3, 4, 5)).filledWithIndex()
            print(volume.formatted((2,0)))
            
            let subView = volume.view(at: [1, 1, 1], extents: [2, 2, 2])
            print(subView.formatted((2,0)))
            
            let subViewSum = sum(subView).scalarValue()
            XCTAssert(subViewSum == 312)
        }
        
        //--------------------------------
        // repeat a value
        // No matter the extents, `volume` only uses the shared storage
        // from `value` and repeats it through indexing
        do {
            let volume = Volume<Int32>((2, 3, 10), repeating: Volume(42))
            print(volume.formatted((2,0)))
        }
        
        //--------------------------------
        // repeat a vector
        // No matter the extents, `matrix` only uses the shared storage
        // from `rowVector` and repeats it through indexing
        do {
            let rowVector = Matrix<Int32>((1, 10), sequence: 0..<10)
            let rmatrix = Matrix((10, 10), repeating: rowVector)
            print(rmatrix.formatted((2,0)))

            let colVector = Matrix<Int32>((10, 1), sequence: 0..<10)
            let cmatrix = Matrix((10, 10), repeating: colVector)
            print(cmatrix.formatted((2,0)))
        }
    }
}
