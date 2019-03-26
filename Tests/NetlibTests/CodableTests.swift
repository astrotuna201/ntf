//
//  CodableTests.swift
//  NetlibTests
//
//  Created by Edward Connell on 3/23/19.
//
import XCTest
import Foundation
import TensorFlow
@testable import Netlib
@testable import DeepLearning

class CodableTests: XCTestCase {
    static var allTests = [
        ("test_PrimaryOps", test_PrimaryOps),
        ("test_ShapeCodable", test_ShapeCodable),
    ]

    func test_ShapeCodable() {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct results.
        let shape = DataShape(64, 100)
        
        do {
            let jsonData = try JSONEncoder().encode(shape)
            let jsonString = String(data: jsonData, encoding: .utf8)!
            print(jsonString)
            print(jsonString.utf8.count)
        } catch {
            print(String(describing: error))
        }
    }
    
    func test_PrimaryOps() {
        let a = TensorView<Float>(scalars: [1, 2, 3, 4])
        let b = TensorView<Float>(scalars: [4, 3, 2, 1])
        let c = a + b
        XCTAssert(c == TensorView<Float>(scalars: [5, 5, 5, 5]))
    }
}
