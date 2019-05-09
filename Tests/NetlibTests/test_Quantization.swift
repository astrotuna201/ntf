//******************************************************************************
//  Created by Edward Connell on 5/8/19
//  Copyright © 2016 Connell Research. All rights reserved.
//
import XCTest
import Foundation

@testable import Netlib

class test_Quantization: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_perfMatrix", test_perfMatrix),
    ]

    //==========================================================================
    // test_perfMatrix
    // initializes two matrices and adds them together
    func test_perfMatrix() {
//        do {
            Platform.log.level = .diagnostic
            let rows = 256
            let cols = 256
            let scalars = [UInt8](repeating: 127, count: (rows * cols))

            var qmatrix = QMatrix<UInt8, Float>((rows, cols), scalars: scalars)
            qmatrix.scale = 1 / Float(UInt8.max)

            measure {
                do {
                    _ = try qmatrix.values()
//                    let values = try qmatrix.array()
                    print("hi")
                } catch {
                    XCTFail(String(describing: error))
                }
            }
//        } catch {
//            XCTFail(String(describing: error))
//        }
    }
}
