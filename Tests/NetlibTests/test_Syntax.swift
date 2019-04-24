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
//        // initialize a matrix with a sequence and take the sum
//        // use shortcut syntax for extents
//        let matrix = Matrix<Float>(3, 5, sequence: 0..<15)
//        let sum1 = matrix.sum().scalarValue()
//        XCTAssert(sum1 == 105.0)
        
        // - initialize a volume using explicit extents
        // - fill with indexes on the default device
        // - take the sum of the sub view on the device
        // - return the scalar value back to the app thread
        let volume = Volume<Int32>(extents: [3, 4, 5]).filledWithIndex()
//        print(volume.formatted(scalarFormat: (2,0)))
        
        let subView = volume.view(at: [1, 1, 1], extents: [2, 2, 2])
        print(subView.formatted(scalarFormat: (2,0)))
        
        let sum2 = sum(subView).scalarValue()
        XCTAssert(sum2 == 105)
    }
}
