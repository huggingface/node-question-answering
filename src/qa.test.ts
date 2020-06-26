import * as tf from "@tensorflow/tfjs-node";
import { mocked } from "ts-jest";

import { QAClient } from "./";
import { initModel } from "./models";
import { RuntimeType } from "./runtimes";
import { Tokenizer } from "./tokenizers";

const basicQuestion = "What was the final score?";
const basicContext = `
  Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season.
  The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California.
  As this was the 50th Super Bowl, the league emphasized the "golden anniversary" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as "Super Bowl L"), so that the logo could prominently feature the Arabic numerals 50.
`;

// const basicContext = `
// At his father's death on 16 September 1380, Charles VI inherited the throne of France. His coronation took place on 4 November 1380, at Reims Cathedral. Charles VI was only 11 years old when he was crowned King of France. During his minority, France was ruled by Charles' uncles, as regents. Although the royal age of majority was 14 (the "age of accountability" under Roman Catholic canon law), Charles terminated the regency only at the age of 21.
// The regents were Philip the Bold, Duke of Burgundy, Louis I, Duke of Anjou, and John, Duke of Berry – all brothers of Charles V – along with Louis II, Duke of Bourbon, Charles VI's maternal uncle. Philip took the dominant role during the regency. Louis of Anjou was fighting for his claim to the Kingdom of Naples after 1382, dying in 1384; John of Berry was interested mainly in the Languedoc, and not particularly interested in politics; and Louis of Bourbon was a largely unimportant figure, owing to his personality (showing signs of mental instability) and status (since he was not the son of a king).
// During the rule of his uncles, the financial resources of the kingdom, painstakingly built up by his father, were squandered for the personal profit of the dukes, whose interests were frequently divergent or even opposing. During that time, the power of the royal administration was strengthened and taxes re-established. The latter policy represented a reversal of the deathbed decision of the king's father Charles V to repeal taxes, and led to tax revolts, known as the Harelle. Increased tax revenues were needed to support the self-serving policies of the king's uncles, whose interests were frequently in conflict with those of the crown and with each other. The Battle of Roosebeke (1382), for example, brilliantly won by the royal troops, was prosecuted solely for the benefit of Philip of Burgundy. The treasury surplus carefully accumulated by Charles V was quickly squandered.
// Charles VI brought the regency to an end in 1388, taking up personal rule. He restored to power the highly competent advisors of Charles V, known as the Marmousets, who ushered in a new period of high esteem for the crown. Charles VI was widely referred to as Charles the Beloved by his subjects.
// He married Isabeau of Bavaria on 17 July 1385, when he was 17 and she 14 (and considered an adult at the time). Isabeau had 12 children, most of whom died young. Isabeau's first child, named Charles, was born in 1386, and was Dauphin of Viennois (heir apparent), but survived only 3 months. Her second child, Joan, was born on 14 June 1388, but died in 1390. Her third child, Isabella, was born in 1389. She was married to Richard II, King of England in 1396, at the age of 6, and became Queen of England. Richard died in 1400 and they had no children. Richard's successor, Henry IV, wanted Isabella then to marry his son, 14-year-old future king Henry V, but she refused. She was married again in 1406, this time to her cousin, Charles, Duke of Orléans, at the age of 17. She died in childbirth at the age of 19.
// Isabeau's fourth child, Joan, was born in 1391, and was married to John VI, Duke of Brittany in 1396, at an age of 5; they had children. Isabeau's fifth child born in 1392 was also named Charles, and was Dauphin. The young Charles was betrothed to Margaret of Burgundy in 1396, but died at the age of 9. Isabeau's sixth child, Mary, was born in 1393. She was never married, and had no children. Isabeau's seventh child, Michelle, was born in 1395. She was engaged to Philip, son of John the Fearless, Duke of Burgundy, in 1404 (both were then aged 8) and they were married in 1409, aged 14. She had one child who died in infancy, before she died in 1422, aged 27.
// Isabeau's eighth child, Louis, was born in 1397, and was also Dauphin. He married Margaret of Burgundy, who had previously been betrothed to his brother Charles. The marriage produced no children by the time of Louis's death in 1415, aged 18.
// Isabeau's ninth child, John, was born in 1398, and was also Dauphin from 1415, after the death of his brother Louis. He was married to Jacqueline, Countess of Hainaut in 1415, then aged 17, but they did not have any children before he died in 1417, aged 19. Isabeau's tenth child, Catherine, was born in 1401. She was married firstly to Henry V, King of England in 1420, and they had one child, who became Henry VI of England. Henry V died suddenly in 1422. Catherine may then have secretly married Owen Tudor in 1429 and had additional children, including Edmund Tudor, the father of Henry VII. She died in 1437, aged 36.
// `;
// const basicQuestion = `Who did Charles VI marry?`;

(async () => {
  const qaClientOne = await QAClient.fromOptions();
  const model = await initModel({ name: "deepset/roberta-base-squad2" });
  const qaClientTwo = await QAClient.fromOptions({ model });

  await qaClientOne.predict(basicQuestion, basicContext);
  await qaClientTwo.predict(basicQuestion, basicContext);

  console.log("\n\n\n ---- LOADED ---- \n\n\n");
  const time = Date.now();

  // const predOne = await qaClientOne.predict(basicQuestion, basicContext);
  // console.log(predOne);
  // const predTwo = await qaClientTwo.predict(basicQuestion, basicContext);
  // console.log(predTwo);
  // console.log(Date.now() - time);

  // qaClientTwo
  //     .predict(basicQuestion, basicContext)
  //     .then(r => console.log(r, 2, Date.now() - time));
  // qaClientOne
  //     .predict(basicQuestion, basicContext)
  //     .then(r => console.log(r, 1, Date.now() - time));

  for (let index = 0; index < 30; index++) {
    for (const qaClient of [qaClientOne, qaClientTwo]) {
      qaClient
        .predict(basicQuestion, basicContext)
        .then(r => console.log(index, r, Date.now() - time, qaClient.modelName));
      // qaClient
      //   .predict("What was the final score?", basicContext)
      //   .then(r => console.log(r, Date.now() - time, qaClient.modelName));
      // qaClient
      //   .predict("When was the game played?", basicContext)
      //   .then(r => console.log(r, Date.now() - time, qaClient.modelName));
    }
  }
  // console.log("final num models", tf.node.getNumOfSavedModels());
  // await new Promise(r => setTimeout(r, 10000));
  // console.log("TEEEEEEEEEST AVAIABILITY\n\n\n\n\n");
  // models_1.initModel({ name: "distilbert-base-uncased-distilled-squad" });
  // models_1.initModel({ name: "twmkn9/bert-base-uncased-squad2" });
})();

// describe("QAClient", () => {
//   describe("fromOptions", () => {
//     // eslint-disable-next-line jest/no-disabled-tests
//     it.skip("instantiates a QAClient with custom tokenizer when provided", async () => {
//       const tokenizer = jest.fn();
//       const qaClient = await QAClient.fromOptions({
//         tokenizer: (tokenizer as unknown) as Tokenizer
//       });
//       // eslint-disable-next-line @typescript-eslint/no-explicit-any
//       expect((qaClient as any).tokenizer).toBe(tokenizer);
//     });

//     it("leads to answer without inference time by default", async () => {
//       const qaClient = await QAClient.fromOptions();
//       const predOne = await qaClient.predict(basicQuestion, basicContext);
//       expect(predOne?.inferenceTime).toBeUndefined();
//     });

//     it("leads to answer with inference time when `timeIt` is `true`", async () => {
//       const qaClient = await QAClient.fromOptions({ timeIt: true });
//       const predOne = await qaClient.predict(basicQuestion, basicContext);
//       expect(typeof predOne?.inferenceTime).toBe("number");
//     });
//   });

//   describe("predict", () => {
//     let qa: QAClient;

//     const shorts = [
//       {
//         context: `
//           Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season.
//           The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California.
//           As this was the 50th Super Bowl, the league emphasized the "golden anniversary" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as "Super Bowl L"), so that the logo could prominently feature the Arabic numerals 50.
//         `,
//         question: ["Who won the Super Bowl?", "Denver Broncos"]
//       },
//       {
//         context: `
//           One of the most famous people born in Warsaw was Maria Skłodowska-Curie, who achieved international recognition for her research on radioactivity and was the first female recipient of the Nobel Prize. Famous musicians include Władysław Szpilman and Frédéric Chopin. Though Chopin was born in the village of Żelazowa Wola, about 60 km (37 mi) from Warsaw, he moved to the city with his family when he was seven months old. Casimir Pulaski, a Polish general and hero of the American Revolutionary War, was born here in 1745.
//         `,
//         question: ["Where was Chopin born?", "Żelazowa Wola"]
//       }
//     ];

//     const long = {
//       context: `
//         At his father's death on 16 September 1380, Charles VI inherited the throne of France. His coronation took place on 4 November 1380, at Reims Cathedral. Charles VI was only 11 years old when he was crowned King of France. During his minority, France was ruled by Charles' uncles, as regents. Although the royal age of majority was 14 (the "age of accountability" under Roman Catholic canon law), Charles terminated the regency only at the age of 21.

//         The regents were Philip the Bold, Duke of Burgundy, Louis I, Duke of Anjou, and John, Duke of Berry – all brothers of Charles V – along with Louis II, Duke of Bourbon, Charles VI's maternal uncle. Philip took the dominant role during the regency. Louis of Anjou was fighting for his claim to the Kingdom of Naples after 1382, dying in 1384; John of Berry was interested mainly in the Languedoc, and not particularly interested in politics; and Louis of Bourbon was a largely unimportant figure, owing to his personality (showing signs of mental instability) and status (since he was not the son of a king).

//         During the rule of his uncles, the financial resources of the kingdom, painstakingly built up by his father, were squandered for the personal profit of the dukes, whose interests were frequently divergent or even opposing. During that time, the power of the royal administration was strengthened and taxes re-established. The latter policy represented a reversal of the deathbed decision of the king's father Charles V to repeal taxes, and led to tax revolts, known as the Harelle. Increased tax revenues were needed to support the self-serving policies of the king's uncles, whose interests were frequently in conflict with those of the crown and with each other. The Battle of Roosebeke (1382), for example, brilliantly won by the royal troops, was prosecuted solely for the benefit of Philip of Burgundy. The treasury surplus carefully accumulated by Charles V was quickly squandered.

//         Charles VI brought the regency to an end in 1388, taking up personal rule. He restored to power the highly competent advisors of Charles V, known as the Marmousets, who ushered in a new period of high esteem for the crown. Charles VI was widely referred to as Charles the Beloved by his subjects.

//         He married Isabeau of Bavaria on 17 July 1385, when he was 17 and she 14 (and considered an adult at the time). Isabeau had 12 children, most of whom died young. Isabeau's first child, named Charles, was born in 1386, and was Dauphin of Viennois (heir apparent), but survived only 3 months. Her second child, Joan, was born on 14 June 1388, but died in 1390. Her third child, Isabella, was born in 1389. She was married to Richard II, King of England in 1396, at the age of 6, and became Queen of England. Richard died in 1400 and they had no children. Richard's successor, Henry IV, wanted Isabella then to marry his son, 14-year-old future king Henry V, but she refused. She was married again in 1406, this time to her cousin, Charles, Duke of Orléans, at the age of 17. She died in childbirth at the age of 19.

//         Isabeau's fourth child, Joan, was born in 1391, and was married to John VI, Duke of Brittany in 1396, at an age of 5; they had children. Isabeau's fifth child born in 1392 was also named Charles, and was Dauphin. The young Charles was betrothed to Margaret of Burgundy in 1396, but died at the age of 9. Isabeau's sixth child, Mary, was born in 1393. She was never married, and had no children. Isabeau's seventh child, Michelle, was born in 1395. She was engaged to Philip, son of John the Fearless, Duke of Burgundy, in 1404 (both were then aged 8) and they were married in 1409, aged 14. She had one child who died in infancy, before she died in 1422, aged 27.

//         Isabeau's eighth child, Louis, was born in 1397, and was also Dauphin. He married Margaret of Burgundy, who had previously been betrothed to his brother Charles. The marriage produced no children by the time of Louis's death in 1415, aged 18.

//         Isabeau's ninth child, John, was born in 1398, and was also Dauphin from 1415, after the death of his brother Louis. He was married to Jacqueline, Countess of Hainaut in 1415, then aged 17, but they did not have any children before he died in 1417, aged 19. Isabeau's tenth child, Catherine, was born in 1401. She was married firstly to Henry V, King of England in 1420, and they had one child, who became Henry VI of England. Henry V died suddenly in 1422. Catherine may then have secretly married Owen Tudor in 1429 and had additional children, including Edmund Tudor, the father of Henry VII. She died in 1437, aged 36.
//       `,
//       questions: [
//         ["When did his father die?", "16 September 1380"],
//         ["Who did Charles VI marry?", "Isabeau of Bavaria"],
//         ["What was the name of Isabeau's tenth child?", "Catherine"]
//       ]
//     };

//     describe("using SavedModel format", () => {
//       beforeAll(async () => {
//         qa = await QAClient.fromOptions();
//       }, 100000000);

//       it.each(shorts)("gives the correct answer with short contexts", async short => {
//         const result = await qa.predict(short.question[0], short.context);
//         expect(result?.text).toEqual(short.question[1]);
//       });

//       for (const question of long.questions) {
//         it("gives the correct answer with long contexts", async () => {
//           const result = await qa.predict(question[0], long.context);
//           expect(result?.text).toEqual(question[1]);
//         });
//       }
//     });

//     describe("using TFJS format", () => {
//       beforeAll(async () => {
//         const model = await initModel({
//           name: "distilbert-base-cased-distilled-squad",
//           runtime: RuntimeType.TFJS
//         });

//         qa = await QAClient.fromOptions({ model });
//       }, 100000000);

//       it.each(shorts)("gives the correct answer with short contexts", async short => {
//         const result = await qa.predict(short.question[0], short.context);
//         expect(result?.text).toEqual(short.question[1]);
//       });

//       for (const question of long.questions) {
//         it("gives the correct answer with long contexts", async () => {
//           const result = await qa.predict(question[0], long.context);
//           expect(result?.text).toEqual(question[1]);
//         });
//       }
//     });
//   });
// });
