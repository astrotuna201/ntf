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
        ("test_matrix", test_matrix),
        ("test_RGBImage", test_RGBImage),
    ]
    
    //==========================================================================
    // test_vector
    // encodes and decodes
    func test_vector() {
        do {
            let jsonEncoder = JSONEncoder()
            let expected = stride(from: -2.0, to: 2.0, by: 0.5).map { Float($0)}
            let vector = Vector<Float>(elements: expected)
            let jsonData = try jsonEncoder.encode(vector)
//            let jsonVectorString = String(data: jsonData, encoding: .utf8)!
//            print(jsonVectorString)
            let decoder = JSONDecoder()
            let vector2 = try decoder.decode(Vector<Float>.self, from: jsonData)
            let values = try vector2.array()
            XCTAssert(values == expected)
        } catch {
            XCTFail(String(describing: error))
        }
    }

    //==========================================================================
    // test_matrix
    // encodes and decodes
    func test_matrix() {
        do {
            let jsonEncoder = JSONEncoder()
            let expected = (0..<10).map { Float($0) }
            let matrix = Matrix<Float>((2, 5), elements: expected)
            let jsonData = try jsonEncoder.encode(matrix)
//            let jsonVectorString = String(data: jsonData, encoding: .utf8)!
//            print(jsonVectorString)
            let decoder = JSONDecoder()
            let matrix2 = try decoder.decode(Matrix<Float>.self, from: jsonData)
            let values = try matrix2.array()
            XCTAssert(values == expected)
        } catch {
            XCTFail(String(describing: error))
        }
    }

    //==========================================================================
    // test_RGBImage
    // encodes and decodes
    func test_RGBImage() {
        do {
            typealias Image = Matrix<RGB<Float>>
            let jsonEncoder = JSONEncoder()
            let pixels = [RGB<Float>(0, 0.5, 1), RGB<Float>(0.25, 0.5, 0.75)]
            let image = Image((1, 2), name: "pixels", elements: pixels)
            let jsonData = try jsonEncoder.encode(image)
            //            let jsonVectorString = String(data: jsonData, encoding: .utf8)!
            //            print(jsonVectorString)
            let decoder = JSONDecoder()
            let image2 = try decoder.decode(Image.self, from: jsonData)
            let values = try image2.array()
            XCTAssert(values == pixels)
        } catch {
            XCTFail(String(describing: error))
        }
    }
}
