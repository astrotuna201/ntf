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
        ("test_vector", test_vector),
    ]
    
    //==========================================================================
    // test_TensorArray
    // initializes two matrices and adds them together
    func test_vector() {
        do {
            let jsonEncoder = JSONEncoder()
            let expected = stride(from: -2.0, to: 2.0, by: 0.5).map { Float($0) }
            let vector = Vector<Float>(elements: expected)
            let jsonData = try jsonEncoder.encode(vector)
            let jsonVectorString = String(data: jsonData, encoding: .utf8)!
            print(jsonVectorString)

            let decoder = JSONDecoder()
            let vector2 = try decoder.decode(Vector<Float>.self, from: jsonData)
            let values = try vector2.array()
            XCTAssert(values == expected)
        } catch {
            XCTFail(String(describing: error))
        }
    }

}
