//******************************************************************************
//  Created by Edward Connell on 1/24/17
//  Copyright © 2017 Connell Research. All rights reserved.
//
import Dispatch

//==============================================================================
// Mutex
public final class Mutex {
    // properties
    private let semaphore = DispatchSemaphore(value: 1)

    // functions
    func sync<R>(execute work: () throws -> R) rethrows -> R {
        semaphore.wait()
        defer { semaphore.signal() }
        return try work()
    }
}

//==============================================================================
// AtomicCounter
public final class AtomicCounter {
    // properties
    private var counter: Int
    private let mutex = Mutex()
    
    public var value: Int {
        get { return mutex.sync { counter } }
        set { return mutex.sync { counter = newValue } }
    }

    // initializers
	public init(value: Int = 0) { counter = value }

    // functions
	public func increment() -> Int {
		return mutex.sync {
			counter += 1
			return counter
		}
	}
}
