require('dotenv').config()
const run = require("@xmtp/bot-starter").default;

// const { Wallet } = require("ethers");

// const wallet = Wallet.createRandom();
// console.log(wallet)
// console.log("Set your environment variable: KEY=" + wallet.privateKey);

// Call `run` with a handler function. The handler function is called
// with a HandlerContext
run(async (context) => {
  // When someone sends your bot a message, you can get the DecodedMessage
  // from the HandlerContext's `message` field
  const messageBody = context.message.content;
  // console.log(messageBody);

  const body = {
    prompt: messageBody
  };

  const response = await fetch(process.env.COPILOT_MODEL_SERVER_URL, {
    method: 'post',
    body: JSON.stringify(body),
    headers: {'Content-Type': 'application/json'}
  });
  // console.log(await response)
  const data = await response.json();

  console.log(data);

  const res = data["web3_devrel_response"]

  // To reply, just call `reply` on the HandlerContext.
  await context.reply(res);
});

