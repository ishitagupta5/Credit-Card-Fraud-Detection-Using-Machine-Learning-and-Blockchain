// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * FraudLog
 * --------
 * Stores a hash for every transaction the ML model flags as fraudulent.
 * Kept minimal (no storage) so deployment is instant and gas‑cheap.
 */
contract FraudLog {
    event Logged(bytes32 indexed txHash, bool isFraud);

    /// @param txHash 32‑byte hash of the transaction (e.g., sha256(row‑ID))
    /// @param isFraud always `true` in this demo, but lets you log "clean" too
    function log(bytes32 txHash, bool isFraud) external {
        emit Logged(txHash, isFraud);
    }
}
