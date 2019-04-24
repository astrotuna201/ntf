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
            let matrix = Matrix<Float>(3, 5, sequence: 0..<15)
            let sum = matrix.sum().scalarValue()
            XCTAssert(sum == 105.0)
        }
        
        //--------------------------------
        // Select and sum a 3D sub region
        // - initialize a volume using explicit extents
        // - fill with indexes on the default device
        // - take the sum of the sub view on the device
        // - return the scalar value back to the app thread
        do {
            let volume = Volume<Int32>(extents: [3, 4, 5]).filledWithIndex()
            print(volume.formatted(scalarFormat: (2,0)))
            let sample = volume.view(at: [1, 1, 1], extents: [2, 2, 2])
            print(sample.formatted(scalarFormat: (2,0)))
            let sampleSum = sum(sample).scalarValue()
            XCTAssert(sampleSum == 312)
        }
        
        //--------------------------------
        // repeat a value
        // No matter the extents, `volume` only uses the shared storage
        // from `value` and repeats it through indexing
        do {
            let volume = Volume<Int32>(extents: [2, 3, 10],
                                       repeating: Volume(42))
            print(volume.formatted(scalarFormat: (2,0)))
        }
        
        //--------------------------------
        // repeat a vector
        // No matter the extents, `matrix` only uses the shared storage
        // from `rowVector` and repeats it through indexing
        do {
            let rowVector = Matrix<Int32>(1, 10, sequence: 0..<10)
            let matrix = Matrix(extents: [10, 10], repeating: rowVector)
            print(matrix.formatted(scalarFormat: (2,0)))
        }
    }
}
