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
    // test_StreamEventWait
    func test_StreamEventWait() {
        do {
            Platform.log.level = .diagnostic
            Platform.local.log.categories = [.streamSync]
            
            let stream = Platform.local.createStream(serviceName: "cpuUnitTest")
            let event = try stream.createEvent(options: StreamEventOptions())
            try stream.debugDelay(seconds: 0.001)
            try stream.wait(for: stream.record(event: event))
            XCTAssert(event.occurred, "wait failed to block")
        } catch {
            XCTFail(String(describing: error))
        }
    }
    
    //==========================================================================
    // test_perfCreateStreamEvent
    func test_perfCreateStreamEvent() {
        let stream = Platform.local.createStream()
        #if !DEBUG
        self.measure {
            do {
                for _ in 0..<1000 {
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
    func test_perfRecordStreamEvent() {
        #if !DEBUG
        let stream = Platform.local.createStream()
        var events = [StreamEvent]()
        do {
            for _ in 0..<10000 {
                try events.append(
                    stream.createEvent(options: StreamEventOptions()))
            }
        } catch {
            XCTFail(String(describing: error))
        }

        self.measure {
            do {
                for event in events {
                    _ = try stream.record(event: event)
                }
                try stream.waitUntilStreamIsComplete()
            } catch {
                XCTFail(String(describing: error))
            }
        }
        #endif
    }
}
