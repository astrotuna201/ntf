//******************************************************************************
// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
import Dispatch
import Foundation

#if os(Linux)
import Glibc
#endif

public protocol ObjectTracking: class {
	var trackingId: Int { get }
}

//==============================================================================
/// ObjectTracker
/// The object tracker class is used during debug to track the lifetime of
/// class objects to prevent reference cycles and memory leaks
final public class ObjectTracker {
	//--------------------------------------------------------------------------
	// types
	public struct ItemInfo {
        var isStatic: Bool
        let namePath: String?
		let supplementalInfo: String?
        let typeName: String
	}

    // singleton
    private init() {}
    
	//--------------------------------------------------------------------------
	// properties
	/// global shared instance
	public static let global = ObjectTracker()
    /// thread access queue
	private let queue = DispatchQueue(label: "ObjectTracker.queue")
    /// counter to generate object ids
	public let counter = AtomicCounter()
    /// init registration tracking id to break on
	public var debuggerRegisterBreakId = -1
    /// deinit break id
	public var debuggerRemoveBreakId = -1
    // the list of currently registered objects
	public private(set) var activeObjects = [Int: ItemInfo]()
    /// true if there are currently unreleased non static objects
	public var hasUnreleasedObjects: Bool {
        return activeObjects.first { !$0.value.isStatic } != nil
    }

	//--------------------------------------------------------------------------
	// getActiveObjectReport
	public func getActiveObjectReport(includeStatics: Bool = false) -> String {
		var result = "\n"
		var activeCount = 0
		for objectId in (activeObjects.keys.sorted { $0 < $1 }) {
			let info = activeObjects[objectId]!
			if includeStatics || !info.isStatic {
				result += getObjectDescription(trackingId: objectId,
                                               info: info) + "\n"
				activeCount += 1
			}
		}
		if activeCount > 0 {
			result += "\nObjectTracker contains \(activeCount) live objects\n"
		}
		return result
	}

    //--------------------------------------------------------------------------
	// getObjectDescription
	private func getObjectDescription(trackingId: Int, info: ItemInfo) -> String {
        return "[\(info.typeName)(\(trackingId))" +
            (info.supplementalInfo == nil ? "]":" \(info.supplementalInfo!)]") +
            (info.namePath == nil ? "" : " path: \(info.namePath!)")
	}

	//--------------------------------------------------------------------------
	// register(object:
    public func register(_ object: ObjectTracking,
                         namePath: String? = nil,
                         supplementalInfo: @autoclosure () -> String? = nil,
                         isStatic: Bool = false) -> Int {
        #if DEBUG
        let info = ItemInfo(isStatic: isStatic,
                            namePath: namePath,
                            supplementalInfo: supplementalInfo(),
                            typeName: String(describing: object.self))
        let trackingId = counter.increment()

        register(trackingId: trackingId, info: info)
        return trackingId
        
        #else
        return counter.increment()
        #endif
    }

    //--------------------------------------------------------------------------
    // register
    private func register(trackingId: Int, info: ItemInfo) {
        queue.sync {
            if trackingId == debuggerRegisterBreakId {
                print("ObjectTracker debug break for id(\(trackingId))")
                raise(SIGINT)
            }
            activeObjects[trackingId] = info
        }
    }

	//--------------------------------------------------------------------------
	// remove
	public func remove(trackingId: Int) {
		#if DEBUG
        _ = queue.sync {
            if trackingId == debuggerRemoveBreakId {
                print("ObjectTracker debug break remove for id(\(trackingId))")
                raise(SIGINT)
            }
            activeObjects.removeValue(forKey: trackingId)
        }
		#endif
	}
}
