// scripts/deploy.js
async function main() {
    const FraudLog = await ethers.getContractFactory("FraudLog");
    const contract = await FraudLog.deploy();
    await contract.deployed();
    console.log("FraudLog deployed →", contract.address);
  }
  main().catch((error) => { console.error(error); process.exitCode = 1; });
  