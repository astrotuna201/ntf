//******************************************************************************
//  Created by Edward Connell on 5/23/19
//  Copyright © 2019 Connell Research. All rights reserved.
//
import XCTest
import Foundation

@testable import Netlib

class test_Codable: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_TensorArray", test_TensorArray),
    ]
    
    //==========================================================================
    // test_TensorArray
    // initializes two matrices and adds them together
    func test_TensorArray() {
        do {
            let jsonEncoder = JSONEncoder()
            let expected = stride(from: -2.0, to: 2.0, by: 0.5).map { Float($0) }
            let expectedString = try String(data: jsonEncoder.encode(expected),
                                            encoding: .utf8)!

            let vector = Vector<Float>(elements: expected)
            let jsonData = try jsonEncoder.encode(vector.tensorArray)
            let jsonVectorString = String(data: jsonData, encoding: .utf8)!
            print(jsonVectorString)
            XCTAssert(jsonVectorString == expectedString)
        } catch {
            XCTFail(String(describing: error))
        }
    }

}
