//
//  CodableTests.swift
//  NetlibTests
//
//  Created by Edward Connell on 3/23/19.
//
import XCTest
import Foundation
@testable import Netlib

class test_DataShape: XCTestCase {
    static var allTests = [
        ("test_squeezed", test_squeezed),
        ("test_transposed", test_transposed),
    ]

    func test_squeezed() {
        XCTAssert(DataShape(10, 1, 4, 3, 1).squeezed().extents == [10,4,3])
        XCTAssert(DataShape(10, 1, 4, 3, 1, 1).squeezed().extents == [10,4,3])
        XCTAssert(DataShape(1, 1, 4, 3, 1).squeezed().extents == [4,3])
        XCTAssert(DataShape(1, 1, 4, 3, 5).squeezed().extents == [4,3,5])
        XCTAssert(DataShape(1, 1, 4, 1, 1, 3, 5).squeezed().extents == [4,3,5])
        
        XCTAssert(DataShape(10, 1, 4, 3, 1).squeezed(axes: [0,4]).extents == [10,1,4,3])
        XCTAssert(DataShape(10, 1, 4, 3, 1, 1).squeezed(axes: [0,5]).extents == [10,1,4,3,1])
        XCTAssert(DataShape(1, 1, 4, 3, 1).squeezed(axes: [1,3]).extents == [1,4,3,1])
        XCTAssert(DataShape(1, 1, 4, 3, 5).squeezed(axes: [3,3]).extents == [1,1,4,3,5])
        XCTAssert(DataShape(1, 1, 4, 1, 1, 3, 5).squeezed(axes: []).extents == [1, 1, 4, 1, 1, 3, 5])
    }
    
    func test_transposed() {
//        let avals = (0..<6).map { Float($0) }
//        let a = TensorView<Float>(extents: 2,3, scalars: avals)
        
    }
}
