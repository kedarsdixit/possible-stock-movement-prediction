import React from "react";
import Chart from "./Chart";
import "./App.css";
import { csvParse } from "d3-dsv";
import { timeParse } from "d3-time-format";
import MACD from "./MACD";
import RSI from "./RSI";
import MA from "./MA";
import Stochastics from "./Stochastics";
import BollingerBand from "./BollingerBand";

export default class TI extends React.Component {
  constructor(props) {
    super(props);
  }

  callingFunction = (event) => {
    this.getData().then((data) => {
      this.setState({ data });
    });
  };
  parseData(parse) {
    return function (d) {
      d.date = parse(d.Date);
      d.open = +d.Open;
      d.high = +d.High;
      d.low = +d.Low;
      d.close = +d.Close;
      d.volume = +d.Volume;

      return d;
    };
  }
  getData() {
    //const data={data:this.props.formData}
    //console.log("hiii");
    //{
    //console.log(this.props.txt1);
    //}
    const companyname = this.props.txt1;
    const parseDate = timeParse("%Y-%m-%d");
    const promiseMSFT = fetch(`${companyname}.csv`)
      .then((response) => response.text())
      .then((data) => csvParse(data, this.parseData(parseDate)));
    return promiseMSFT;
  }

  render() {
    if (this.props.txt1 != "" && this.props.txt2 != "") {
      {
        this.callingFunction();
      }
    }

    /*if (this.props.formData!= null) {
        
            {this.callingFunction()}
        }*/
    if (this.state == null) {
      return <div>...</div>;
    }
    const rendercharts = () => {
      switch (this.props.txt2) {
        case "MACD":
          return <MACD data={this.state.data} />;
        //break;
        case "RSI":
          return <RSI data={this.state.data} />;
        //break;
        case "Candelstick chart":
          return <Chart data={this.state.data} />;
        case "Moving Average":
          return <MA data={this.state.data} />;
        case "Stochastics":
          return <Stochastics data={this.state.data} />;
        case "Bollinger Bands":
          return <BollingerBand data={this.state.data} />;
      }
    };
    return (
      <div className="content">
        <h3>Graph</h3>
        {rendercharts()}
      </div>
    );
  }
}
