//******************************************************************************
//  Created by Edward Connell on 4/26/19
//  Copyright © 2016 Connell Research. All rights reserved.
//
import XCTest
import Foundation

@testable import Netlib

class test_Async: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_StreamEventWait", test_StreamEventWait),
        ("test_perfCreateStreamEvent", test_perfCreateStreamEvent),
    ]

    //==========================================================================
    // test_tensorReferenceBufferSync
    func test_tensorReferenceBufferSync() {
    }

    //==========================================================================
    // test_orphanedStreamShutdown
    func test_orphanedStreamShutdown() {
    }
    
    //==========================================================================
    // test_twoStreamInterleave
    func test_twoStreamInterleave() {
        do {
            Platform.log.level = .diagnostic
            //        Platform.local.log.categories = [.streamSync]
            
            // create a named stream on two different discreet devices
            // cpu devices 1 and 2 are discreet memory versions for testing
//            let stream1 = Platform.local
//                .createStream(deviceId: 1, serviceName: "cpuUnitTest")
            
//            let stream2 = Platform.local
//                .createStream(deviceId: 2, serviceName: "cpuUnitTest")
            
            let m1 = Matrix<Int32>((2, 3), sequence: 0..<6)
            let m2 = Matrix<Int32>((2, 3), sequence: 0..<6)
//            let result = using(stream1) { m1 + m2 }
            var result = Matrix<Int32>((2, 3))
            Netlib.add(m1, m2, result: &result)

            let expected: [Int32] = [0, 2, 4, 6, 8, 10]
            let values = try result.array()
            XCTAssert(values == expected)
            
        } catch {
            XCTFail(String(describing: error))
        }
    }
    
    //==========================================================================
    // test_threeStreamInterleave
    func test_threeStreamInterleave() {
        
    }
    
    //==========================================================================
    // test_StreamEventWait
    func test_StreamEventWait() {
        do {
            Platform.log.level = .diagnostic
            Platform.local.log.categories = [.streamSync]
            
            let stream = Platform.local.createStream(serviceName: "cpuUnitTest")
            let event = try stream.createEvent(options: StreamEventOptions())
            try stream.debugDelay(seconds: 0.001)
            try stream.record(event: event).blockingWait()
            XCTAssert(event.occurred, "wait failed to block")
        } catch {
            XCTFail(String(describing: error))
        }
    }
    
    //==========================================================================
    // test_perfCreateStreamEvent
    // measures the event overhead of creating 10,000 events
    func test_perfCreateStreamEvent() {
        #if !DEBUG
        let stream = Platform.local.createStream()
        self.measure {
            do {
                for _ in 0..<10000 {
                    _ = try stream.createEvent(options: StreamEventOptions())
                }
            } catch {
                XCTFail(String(describing: error))
            }
        }
        #endif
    }

    //==========================================================================
    // test_perfRecordStreamEvent
    // measures the event overhead of processing 10,000 tensors
    func test_perfRecordStreamEvent() {
        #if !DEBUG
        let stream = Platform.local.createStream()
        self.measure {
            do {
                for _ in 0..<10000 {
                    _ = try stream.record(event: stream.createEvent())
                }
                try stream.waitUntilStreamIsComplete()
            } catch {
                XCTFail(String(describing: error))
            }
        }
        #endif
    }
}
